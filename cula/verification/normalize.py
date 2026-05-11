"""
Normalize step: reshape a FetchResult into a flat, rule-friendly view.

Rules never touch raw Pydantic/httpx objects — they read from
NormalizedContext only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from cula.verification.fetch import FetchResult


# ---------------------------------------------------------------------------
# Flat data structures the rule engine consumes
# ---------------------------------------------------------------------------


@dataclass
class FileRef:
    """Minimal proof-file reference."""

    cloud_storage_id: str
    file_name: str | None
    mime_type: str | None
    size_bytes: float | None
    is_sensitive: bool


@dataclass
class NormalizedProof:
    """A single proof attached to an event."""

    proof_id: str
    timestamp: datetime
    proof_type: str  # "file" | "tracking_event"
    file_ref: FileRef | None
    object_types: list[str]  # from proofConfigs


@dataclass
class NormalizedEvent:
    """One event node from the sink's lifecycle graph."""

    event_id: str
    event_type: str            # e.g. "pyrolysis", "biomass_creation", …
    event_category: str        # "step-execution" | "delivery-leg"
    created: datetime
    site_id: UUID | None
    sender_site_id: UUID | None
    receiver_site_id: UUID | None
    transport_distance_km: float | None
    location: tuple[float, float] | None   # (lat, lon)
    input_weight_kg: float | None
    output_weight_kg: float | None
    emission_kg_co2e: float
    proofs: list[NormalizedProof]
    predecessor_ids: list[str]


@dataclass
class SiteInfo:
    """Lookup-friendly site record."""

    site_id: UUID
    name: str
    lat: float
    lon: float
    country: str | None


@dataclass
class SeriesInfo:
    """One machine data-point time series with metadata."""

    config_id: UUID
    name: str
    unit: str | None
    machine_id: UUID
    data: list[tuple[datetime, float]]


@dataclass
class NormalizedContext:
    """Everything the rule engine needs — no Pydantic, no httpx."""

    sink_id: UUID
    sink_created: datetime | None
    gross_impact_kg: float | None
    net_impact_kg: float | None
    carbon_capture_site_id: UUID | None

    events: list[NormalizedEvent]
    pyrolysis_window: tuple[datetime, datetime] | None

    sites: dict[UUID, SiteInfo]
    series: dict[UUID, SeriesInfo]

    cloud_storage_ids_seen: set[str] = field(default_factory=set)

    fetch_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_dt(ms_str: str) -> datetime:
    return datetime.fromtimestamp(int(ms_str) / 1000, tz=timezone.utc)


def _sum_container_weight(containers: list | None) -> float | None:
    if not containers:
        return None
    total = 0.0
    for c in containers:
        for amt in c.content:
            total += amt.weightInKg
    return total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def normalize(fetch: FetchResult) -> NormalizedContext:
    """Convert a FetchResult into a NormalizedContext."""

    sink = fetch.sink

    # --- events ------------------------------------------------------------
    events: list[NormalizedEvent] = []
    pyro_timestamps: list[datetime] = []
    all_cs_ids: set[str] = set()

    if sink.eventGraph and sink.eventGraph.nodes:
        for node_id, info in sink.eventGraph.nodes.items():
            ev = info.event
            if ev is None:
                continue

            created = _ms_to_dt(ev.created)

            loc: tuple[float, float] | None = None
            if ev.location:
                loc = (ev.location.lat, ev.location.long)

            proofs: list[NormalizedProof] = []
            for p in ev.proofs:
                file_ref: FileRef | None = None
                if p.fileReference:
                    fr = p.fileReference
                    file_ref = FileRef(
                        cloud_storage_id=fr.cloudStorageId or "",
                        file_name=fr.fileName,
                        mime_type=fr.type,
                        size_bytes=fr.size,
                        is_sensitive=bool(p.isSensitive),
                    )
                    if fr.cloudStorageId:
                        all_cs_ids.add(fr.cloudStorageId)

                proofs.append(NormalizedProof(
                    proof_id=p.id,
                    timestamp=p.timestamp,
                    proof_type=p.type.value,
                    file_ref=file_ref,
                    object_types=[pc.objectType.value for pc in p.proofConfigs],
                ))

            predecessor_ids = [
                link.eventRef for link in (info.links or [])
            ]

            n_ev = NormalizedEvent(
                event_id=node_id,
                event_type=ev.type.value,
                event_category=ev.eventType.value,
                created=created,
                site_id=ev.siteId,
                sender_site_id=ev.senderSiteId,
                receiver_site_id=ev.receiverSiteId,
                transport_distance_km=ev.transportationDistanceInKm,
                location=loc,
                input_weight_kg=_sum_container_weight(ev.input),
                output_weight_kg=_sum_container_weight(ev.output),
                emission_kg_co2e=ev.emissions.value,
                proofs=proofs,
                predecessor_ids=predecessor_ids,
            )
            events.append(n_ev)

            if ev.type.value == "pyrolysis":
                pyro_timestamps.append(created)

    events.sort(key=lambda e: e.created)

    pyrolysis_window: tuple[datetime, datetime] | None = None
    if pyro_timestamps:
        pyrolysis_window = (min(pyro_timestamps), max(pyro_timestamps))

    # --- sites -------------------------------------------------------------
    sites: dict[UUID, SiteInfo] = {}
    for s in sink.sites or []:
        sites[s.siteRef] = SiteInfo(
            site_id=s.siteRef,
            name=s.name,
            lat=s.location.lat,
            lon=s.location.long,
            country=s.location.country,
        )

    # --- time series -------------------------------------------------------
    series: dict[UUID, SeriesInfo] = {}
    for ts in fetch.time_series:
        cfg = ts.config
        if cfg is None or ts.id is None:
            continue
        data_points = [
            (entry.timestamp, entry.value)
            for entry in (ts.data or [])
        ]
        series[ts.id] = SeriesInfo(
            config_id=ts.id,
            name=cfg.name,
            unit=cfg.unit,
            machine_id=cfg.machineId,
            data=data_points,
        )

    # --- sink-level fields -------------------------------------------------
    sink_created: datetime | None = None
    if sink.created:
        try:
            sink_created = _ms_to_dt(sink.created)
        except (ValueError, TypeError):
            pass

    return NormalizedContext(
        sink_id=sink.id,  # type: ignore[arg-type]
        sink_created=sink_created,
        gross_impact_kg=sink.normalisedGrossImpact,
        net_impact_kg=sink.normalisedNetImpact,
        carbon_capture_site_id=sink.carbonCaptureSiteId,
        events=events,
        pyrolysis_window=pyrolysis_window,
        sites=sites,
        series=series,
        cloud_storage_ids_seen=all_cs_ids,
        fetch_errors=list(fetch.errors),
    )
