"""Minimal tests for cula.verification.normalize — no network calls."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from cula.models import (
    EmissionCalculation,
    EmissionCalculationConfig,
    Event,
    EventGraph,
    EventInfo,
    EventProof,
    FileReference,
    Location,
    MachineDpConfig,
    MachineDataInRange,
    MaterialAmount,
    MaterialContainer,
    Material1 as Material,
    ProofConfig,
    Sink,
    Site,
    TimeSeriesEntry,
)
from cula.verification.fetch import FetchResult
from cula.verification.normalize import normalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SITE_ID = uuid4()
MACHINE_ID = uuid4()
DP_ID = uuid4()

_EMISSION = EmissionCalculation(
    id="em-1",
    type="COMPOSITE",
    value=5.5,
    config=EmissionCalculationConfig(
        id="cfg-1", name="root", type="COMPOSITE", expression="0"
    ),
    nodes=[],
)


def _make_event(
    event_type: str,
    created_ms: int,
    *,
    site_id: UUID | None = None,
    proofs: list[EventProof] | None = None,
    input_containers: list[MaterialContainer] | None = None,
    output_containers: list[MaterialContainer] | None = None,
    transport_km: float | None = None,
    sender: UUID | None = None,
    receiver: UUID | None = None,
    location: Location | None = None,
) -> Event:
    event_category = "delivery-leg" if "delivery" in event_type else "step-execution"
    return Event(
        type=event_type,
        eventRef=str(uuid4()),
        eventType=event_category,
        emissions=_EMISSION,
        proofs=proofs or [],
        created=str(created_ms),
        wayPoints=[],
        siteId=site_id,
        senderSiteId=sender,
        receiverSiteId=receiver,
        transportationDistanceInKm=transport_km,
        location=location,
        input=input_containers,
        output=output_containers,
    )


def _make_sink(
    events: dict[str, EventInfo] | None = None,
    *,
    sites: list[Site] | None = None,
    carbon_capture_site_id: UUID | None = SITE_ID,
    gross: float | None = 100.0,
    net: float | None = 80.0,
) -> Sink:
    graph = None
    if events is not None:
        root = next(iter(events))
        graph = EventGraph(root=root, nodes=events)
    return Sink(
        id=uuid4(),
        created="1700000000000",
        carbonCaptureSiteId=carbon_capture_site_id,
        eventGraph=graph,
        normalisedGrossImpact=gross,
        normalisedNetImpact=net,
        sites=sites,
    )


def _make_fetch(sink: Sink, **kwargs: object) -> FetchResult:
    return FetchResult(
        sink=sink,
        machine_ids=kwargs.get("machine_ids", []),  # type: ignore[arg-type]
        dp_configs=kwargs.get("dp_configs", {}),  # type: ignore[arg-type]
        time_series=kwargs.get("time_series", []),  # type: ignore[arg-type]
        documents=kwargs.get("documents", {}),  # type: ignore[arg-type]
        errors=kwargs.get("errors", []),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalizeEmpty:
    def test_empty_graph(self):
        sink = _make_sink(events=None)
        ctx = normalize(_make_fetch(sink))
        print(f"  → events: {ctx.events}")
        print(f"  → pyrolysis_window: {ctx.pyrolysis_window}")
        assert ctx.events == []
        assert ctx.pyrolysis_window is None

    def test_sink_level_fields(self):
        sink = _make_sink(events=None, gross=120.0, net=95.0)
        ctx = normalize(_make_fetch(sink))
        print(f"  → sink_id: {ctx.sink_id}")
        print(f"  → gross: {ctx.gross_impact_kg}  net: {ctx.net_impact_kg}")
        print(f"  → carbon_capture_site_id: {ctx.carbon_capture_site_id}")
        assert ctx.sink_id == sink.id
        assert ctx.gross_impact_kg == 120.0
        assert ctx.net_impact_kg == 95.0
        assert ctx.carbon_capture_site_id == SITE_ID


class TestNormalizeEvents:
    def test_flat_event_list_sorted(self):
        ev_early = _make_event("biomass_creation", 1_600_000_000_000)
        ev_late = _make_event("pyrolysis", 1_700_000_000_000)
        events = {
            "late": EventInfo(event=ev_late),
            "early": EventInfo(event=ev_early),
        }
        ctx = normalize(_make_fetch(_make_sink(events)))
        print(f"  → event types in order: {[e.event_type for e in ctx.events]}")
        print(f"  → timestamps: {[e.created for e in ctx.events]}")
        assert ctx.events[0].event_type == "biomass_creation"
        assert ctx.events[1].event_type == "pyrolysis"
        assert ctx.events[0].created < ctx.events[1].created

    def test_pyrolysis_window(self):
        ev = _make_event("pyrolysis", 1_700_000_000_000)
        ctx = normalize(_make_fetch(_make_sink({"e1": EventInfo(event=ev)})))
        print(f"  → pyrolysis_window: {ctx.pyrolysis_window}")
        assert ctx.pyrolysis_window is not None

    def test_event_weights(self):
        mat = Material(
            id=uuid4(),
            name="biochar",
            description="test",
            type="biochar",
            colors={"darkColor": "#000", "lightColor": "#fff"},
        )
        container = MaterialContainer(
            id=uuid4(),
            created="1700000000000",
            content=[MaterialAmount(weightInKg=42.5, material=mat)],
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000, output_containers=[container])
        ctx = normalize(_make_fetch(_make_sink({"e1": EventInfo(event=ev)})))
        print(f"  → output_weight_kg: {ctx.events[0].output_weight_kg}")
        assert ctx.events[0].output_weight_kg == 42.5

    def test_delivery_fields(self):
        sender, receiver = uuid4(), uuid4()
        ev = _make_event(
            "biomass_delivery",
            1_700_000_000_000,
            sender=sender,
            receiver=receiver,
            transport_km=123.4,
        )
        ctx = normalize(_make_fetch(_make_sink({"e1": EventInfo(event=ev)})))
        ne = ctx.events[0]
        print(f"  → sender: {ne.sender_site_id}  receiver: {ne.receiver_site_id}")
        print(f"  → transport_km: {ne.transport_distance_km}")
        assert ne.sender_site_id == sender
        assert ne.receiver_site_id == receiver
        assert ne.transport_distance_km == 123.4

    def test_predecessor_ids(self):
        from cula.models import AggregatedSinkEventLink

        ev1 = _make_event("biomass_creation", 1_600_000_000_000)
        ev2 = _make_event("pyrolysis", 1_700_000_000_000)
        link = AggregatedSinkEventLink(
            eventRef="e1", percentage=1.0, connectingContainerIds=[], connectionType="direct"
        )
        ctx = normalize(_make_fetch(_make_sink({
            "e2": EventInfo(event=ev2, links=[link]),
            "e1": EventInfo(event=ev1),
        })))
        pyro = [e for e in ctx.events if e.event_type == "pyrolysis"][0]
        print(f"  → predecessor_ids: {pyro.predecessor_ids}")
        assert pyro.predecessor_ids == ["e1"]


class TestNormalizeProofs:
    def test_proof_with_file_ref(self):
        cs_id = str(uuid4())
        proof = EventProof(
            id="p1",
            timestamp=datetime.now(tz=timezone.utc),
            type="file",
            proofConfigs=[
                ProofConfig(id=uuid4(), objectType="material_creation", key="photo"),
            ],
            isSensitive=False,
            fileReference=FileReference(
                cloudStorageId=cs_id, fileName="photo.jpg", type="image/jpeg", size=1024
            ),
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000, proofs=[proof])
        ctx = normalize(_make_fetch(_make_sink({"e1": EventInfo(event=ev)})))
        np = ctx.events[0].proofs[0]
        print(f"  → proof_type: {np.proof_type}")
        print(f"  → file_ref: name={np.file_ref.file_name} mime={np.file_ref.mime_type} size={np.file_ref.size_bytes}")
        print(f"  → object_types: {np.object_types}")
        assert np.file_ref is not None
        assert np.file_ref.cloud_storage_id == cs_id
        assert cs_id in ctx.cloud_storage_ids_seen

    def test_sensitive_proof_excluded_from_ids(self):
        proof = EventProof(
            id="p1",
            timestamp=datetime.now(tz=timezone.utc),
            type="file",
            proofConfigs=[],
            isSensitive=True,
            fileReference=FileReference(cloudStorageId=None),
        )
        ev = _make_event("pyrolysis", 1_700_000_000_000, proofs=[proof])
        ctx = normalize(_make_fetch(_make_sink({"e1": EventInfo(event=ev)})))
        print(f"  → cloud_storage_ids_seen: {ctx.cloud_storage_ids_seen}")
        assert len(ctx.cloud_storage_ids_seen) == 0


class TestNormalizeSites:
    def test_site_lookup(self):
        site = Site(
            siteRef=SITE_ID,
            name="PyroPlant Berlin",
            location=Location(lat=52.52, long=13.405, country="Germany"),
        )
        sink = _make_sink({"e1": EventInfo(event=_make_event("pyrolysis", 1_700_000_000_000))}, sites=[site])
        ctx = normalize(_make_fetch(sink))
        print(f"  → sites: {ctx.sites}")
        assert SITE_ID in ctx.sites
        si = ctx.sites[SITE_ID]
        assert si.name == "PyroPlant Berlin"
        assert si.lat == 52.52
        assert si.country == "Germany"


class TestNormalizeSeries:
    def test_time_series_dict(self):
        ts_dt = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        cfg = MachineDpConfig(id=DP_ID, name="Reactor temp", unit="°C", siteId=SITE_ID, machineId=MACHINE_ID)
        series = MachineDataInRange(
            id=DP_ID,
            config=cfg,
            data=[TimeSeriesEntry(timestamp=ts_dt, value=450.0)],
        )
        sink = _make_sink({"e1": EventInfo(event=_make_event("pyrolysis", 1_700_000_000_000))})
        fetch = _make_fetch(sink, time_series=[series])
        ctx = normalize(fetch)
        print(f"  → series keys: {list(ctx.series.keys())}")
        si = ctx.series[DP_ID]
        print(f"  → name={si.name} unit={si.unit} points={len(si.data)}")
        print(f"  → first point: {si.data[0]}")
        assert si.name == "Reactor temp"
        assert si.unit == "°C"
        assert len(si.data) == 1
        assert si.data[0] == (ts_dt, 450.0)


class TestNormalizeFetchErrors:
    def test_errors_carried_through(self):
        sink = _make_sink(events=None)
        fetch = _make_fetch(sink, errors=["something went wrong"])
        ctx = normalize(fetch)
        print(f"  → fetch_errors: {ctx.fetch_errors}")
        assert ctx.fetch_errors == ["something went wrong"]
