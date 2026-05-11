"""Minimal tests for cula.verification.fetch"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import httpx
import pytest

from cula.models import (
    EmissionCalculation,
    EmissionCalculationConfig,
    Event,
    EventGraph,
    EventInfo,
    EventProof,
    FileReference,
    MachineDpConfig,
    MachineDataInRange,
    Sink,
    TimeSeriesEntry,
)
from cula.verification.fetch import (
    FetchResult,
    _collect_cloud_storage_ids,
    _find_pyrolysis_window,
    fetch_sink_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SITE_ID = uuid4()
MACHINE_ID = uuid4()
DP_ID = uuid4()

_EMISSION = EmissionCalculation(
    id="em-1",
    type="COMPOSITE",
    value=0.0,
    config=EmissionCalculationConfig(
        id="cfg-1", name="root", type="COMPOSITE", expression="0"
    ),
    nodes=[],
)


def _make_event(event_type: str, created_ms: int) -> Event:
    return Event(
        type=event_type,
        eventRef=str(uuid4()),
        eventType="step-execution",
        emissions=_EMISSION,
        proofs=[],
        created=str(created_ms),
        wayPoints=[],
    )


def _make_sink(
    *,
    events: dict[str, EventInfo] | None = None,
    carbon_capture_site_id: UUID | None = SITE_ID,
) -> Sink:
    graph = None
    if events is not None:
        root = next(iter(events))
        graph = EventGraph(root=root, nodes=events)
    return Sink(
        id=uuid4(),
        carbonCaptureSiteId=carbon_capture_site_id,
        eventGraph=graph,
    )


def _make_dp_config(dp_id: UUID = DP_ID) -> MachineDpConfig:
    return MachineDpConfig(
        id=dp_id,
        name="Reactor temperature",
        unit="°C",
        siteId=SITE_ID,
        machineId=MACHINE_ID,
    )


def _mock_client(sink: Sink, **overrides: object) -> MagicMock:
    client = MagicMock()
    client.get_sink.return_value = sink
    client.list_machines.return_value = overrides.get("machines", [MACHINE_ID])
    client.list_machine_data_points.return_value = overrides.get("dp_ids", [DP_ID])
    client.get_machine_data_point.return_value = overrides.get(
        "dp_config", _make_dp_config()
    )
    client.get_machine_data.return_value = overrides.get("time_series", [])
    client.download_document.return_value = overrides.get("doc_bytes", b"%PDF-fake")
    return client


# ---------------------------------------------------------------------------
# _find_pyrolysis_window
# ---------------------------------------------------------------------------


class TestFindPyrolysisWindow:
    def test_returns_none_without_event_graph(self):
        sink = _make_sink(events=None)
        result = _find_pyrolysis_window(sink)
        print(f"  → window (no graph): {result}")
        assert result is None

    def test_returns_none_without_pyrolysis(self):
        ev = _make_event("biomass_creation", 1_700_000_000_000)
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        result = _find_pyrolysis_window(sink)
        print(f"  → window (no pyrolysis): {result}")
        assert result is None

    def test_returns_window_around_pyrolysis(self):
        ts = 1_700_000_000_000
        ev = _make_event("pyrolysis", ts)
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        window = _find_pyrolysis_window(sink)
        print(f"  → window: {window}")
        assert window is not None
        start, end = window
        centre = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        print(f"  → start={start}  centre={centre}  end={end}")
        assert start < centre < end


# ---------------------------------------------------------------------------
# _collect_cloud_storage_ids
# ---------------------------------------------------------------------------


class TestCollectCloudStorageIds:
    def test_empty_graph(self):
        sink = _make_sink(events=None)
        ids = _collect_cloud_storage_ids(sink)
        print(f"  → ids (no graph): {ids}")
        assert ids == []

    def test_skips_sensitive(self):
        proof = EventProof(
            id="p1",
            timestamp=datetime.now(tz=timezone.utc),
            type="file",
            proofConfigs=[],
            isSensitive=True,
            fileReference=FileReference(cloudStorageId="should-skip"),
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000)
        ev.proofs = [proof]
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        ids = _collect_cloud_storage_ids(sink)
        print(f"  → ids (sensitive): {ids}")
        assert ids == []

    def test_collects_non_sensitive(self):
        cs_id = str(uuid4())
        proof = EventProof(
            id="p1",
            timestamp=datetime.now(tz=timezone.utc),
            type="file",
            proofConfigs=[],
            isSensitive=False,
            fileReference=FileReference(cloudStorageId=cs_id),
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000)
        ev.proofs = [proof]
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        ids = _collect_cloud_storage_ids(sink)
        print(f"  → ids (non-sensitive): {ids}")
        assert ids == [cs_id]


# ---------------------------------------------------------------------------
# fetch_sink_data
# ---------------------------------------------------------------------------


class TestFetchSinkData:
    def test_no_capture_site(self):
        sink = _make_sink(carbon_capture_site_id=None)
        client = _mock_client(sink)
        result = fetch_sink_data(client, sink.id)
        print(f"  → errors: {result.errors}")
        print(f"  → machine_ids: {result.machine_ids}")
        assert any("carbonCaptureSiteId" in e for e in result.errors)
        assert result.machine_ids == []

    def test_no_machines(self):
        sink = _make_sink()
        client = _mock_client(sink, machines=[])
        result = fetch_sink_data(client, sink.id)
        print(f"  → errors: {result.errors}")
        assert any("No machines" in e for e in result.errors)

    def test_happy_path(self):
        ts = 1_700_000_000_000
        ev = _make_event("pyrolysis", ts)
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        ts_entry = TimeSeriesEntry(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            value=450.0,
        )
        series = [MachineDataInRange(id=DP_ID, config=_make_dp_config(), data=[ts_entry])]
        client = _mock_client(sink, time_series=series)

        result = fetch_sink_data(client, sink.id)

        print(f"  → sink id: {result.sink.id}")
        print(f"  → machines: {result.machine_ids}")
        print(f"  → dp_configs: {list(result.dp_configs.keys())}")
        print(f"  → time_series count: {len(result.time_series)}")
        print(f"  → errors: {result.errors}")
        assert result.sink is sink
        assert result.machine_ids == [MACHINE_ID]
        assert DP_ID in result.dp_configs
        assert len(result.time_series) == 1
        assert result.errors == []

    def test_http_error_collected(self):
        sink = _make_sink()
        client = _mock_client(sink)
        resp = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        client.list_machines.side_effect = httpx.HTTPStatusError(
            "not found", request=resp.request, response=resp
        )
        result = fetch_sink_data(client, sink.id)
        print(f"  → errors: {result.errors}")
        assert any("list_machines" in e for e in result.errors)

    def test_documents_fetched_when_enabled(self):
        cs_id = str(uuid4())
        proof = EventProof(
            id="p1",
            timestamp=datetime.now(tz=timezone.utc),
            type="file",
            proofConfigs=[],
            isSensitive=False,
            fileReference=FileReference(cloudStorageId=cs_id),
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000)
        ev.proofs = [proof]
        sink = _make_sink(events={"e1": EventInfo(event=ev)})
        client = _mock_client(sink)

        result = fetch_sink_data(client, sink.id, fetch_documents=True)

        print(f"  → documents keys: {list(result.documents.keys())}")
        print(f"  → doc size: {len(result.documents.get(cs_id, b''))} bytes")
        assert cs_id in result.documents
        client.download_document.assert_called_once_with(cs_id)
