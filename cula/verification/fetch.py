"""
Fetch step: pure I/O against CulaClient.

Retrieves a sink with its full context (event graph, sites, materials …),
the machines / data-point configs / time-series for the carbon-capture site,
and optionally the raw bytes of proof documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx

from cula.client import CulaClient
from cula.models import (
    MachineDataInRange,
    MachineDpConfig,
    MachineDpRequest,
    Sink,
)

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Everything retrieved for one sink.  Fields that could not be fetched stay empty."""

    sink: Sink

    machine_ids: list[UUID] = field(default_factory=list)
    dp_configs: dict[UUID, MachineDpConfig] = field(default_factory=dict)
    time_series: list[MachineDataInRange] = field(default_factory=list)

    documents: dict[str, bytes] = field(default_factory=dict)

    errors: list[str] = field(default_factory=list)


def _find_pyrolysis_window(
    sink: Sink,
    margin: timedelta = timedelta(hours=24),
) -> tuple[datetime, datetime] | None:
    """Derive a time window around pyrolysis events in the sink's event graph."""
    if not sink.eventGraph or not sink.eventGraph.nodes:
        return None

    timestamps: list[int] = []
    for info in sink.eventGraph.nodes.values():
        ev = info.event
        if ev is None:
            continue
        if ev.type.value == "pyrolysis":
            try:
                timestamps.append(int(ev.created))
            except (ValueError, TypeError):
                continue

    if not timestamps:
        return None

    earliest = datetime.fromtimestamp(min(timestamps) / 1000, tz=timezone.utc)
    latest = datetime.fromtimestamp(max(timestamps) / 1000, tz=timezone.utc)
    return (earliest - margin, latest + margin)


def _collect_cloud_storage_ids(sink: Sink) -> list[str]:
    """Walk event proofs and return downloadable cloudStorageIds."""
    ids: list[str] = []
    if not sink.eventGraph or not sink.eventGraph.nodes:
        return ids

    for info in sink.eventGraph.nodes.values():
        ev = info.event
        if ev is None:
            continue
        for proof in ev.proofs:
            if proof.isSensitive:
                continue
            ref = proof.fileReference
            if ref and ref.cloudStorageId:
                ids.append(ref.cloudStorageId)
    return ids


def fetch_sink_data(
    client: CulaClient,
    sink_id: UUID | str,
    *,
    time_bucket: str = "1 hour",
    pyrolysis_margin: timedelta = timedelta(hours=24),
    fetch_documents: bool = False,
) -> FetchResult:
    """Fetch all data for *sink_id*.  Errors are logged and collected, never raised."""

    sink = client.get_sink(sink_id)
    result = FetchResult(sink=sink)

    site_id = sink.carbonCaptureSiteId
    if site_id is None:
        result.errors.append("Sink has no carbonCaptureSiteId — cannot fetch machines.")
        return result

    # --- machines ----------------------------------------------------------
    try:
        result.machine_ids = client.list_machines(site_id)
    except httpx.HTTPStatusError as exc:
        result.errors.append(f"list_machines({site_id}): HTTP {exc.response.status_code}")
        return result

    if not result.machine_ids:
        result.errors.append(f"No machines returned for site {site_id}.")
        return result

    # --- data-point configs ------------------------------------------------
    dp_config_ids: list[UUID] = []
    for machine_id in result.machine_ids:
        try:
            dp_ids = client.list_machine_data_points(machine_id)
            dp_config_ids.extend(dp_ids)
        except httpx.HTTPStatusError as exc:
            result.errors.append(
                f"list_machine_data_points({machine_id}): HTTP {exc.response.status_code}"
            )

    for dp_id in dp_config_ids:
        try:
            result.dp_configs[dp_id] = client.get_machine_data_point(dp_id)
        except httpx.HTTPStatusError as exc:
            result.errors.append(
                f"get_machine_data_point({dp_id}): HTTP {exc.response.status_code}"
            )

    # --- time-series -------------------------------------------------------
    window = _find_pyrolysis_window(sink, margin=pyrolysis_margin)
    if window and result.dp_configs:
        requests = [
            MachineDpRequest(
                source=dp_id,
                start=window[0],
                end=window[1],
                timeBucket=time_bucket,
            )
            for dp_id in result.dp_configs
        ]
        try:
            result.time_series = client.get_machine_data(requests)
        except httpx.HTTPStatusError as exc:
            result.errors.append(f"get_machine_data: HTTP {exc.response.status_code}")
    elif not window:
        result.errors.append("No pyrolysis events found — skipped time-series fetch.")

    # --- documents (optional) ----------------------------------------------
    if fetch_documents:
        for cs_id in _collect_cloud_storage_ids(sink):
            try:
                result.documents[cs_id] = client.download_document(cs_id)
            except httpx.HTTPStatusError as exc:
                result.errors.append(
                    f"download_document({cs_id}): HTTP {exc.response.status_code}"
                )

    if result.errors:
        logger.warning(
            "Fetch completed with %d error(s) for sink %s", len(result.errors), sink_id
        )

    return result
