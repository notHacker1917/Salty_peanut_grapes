"""Tests for cula.verification.rules — all 8 rules + DAG traversal."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from cula.verification.normalize import (
    FileRef,
    NormalizedContext,
    NormalizedEvent,
    NormalizedProof,
    SeriesInfo,
    SiteInfo,
)
from cula.verification.rules import (
    CheckResult,
    RuleConfig,
    check_delivery_distance,
    check_file_reuse,
    check_machine_coverage,
    check_mass_balance,
    check_proof_presence,
    check_site_continuity,
    check_temp_plausible,
    check_timeline_order,
    run_rules,
    topo_leaf_to_root,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CFG = RuleConfig()

SITE_A = uuid4()
SITE_B = uuid4()


def _dt(year: int, month: int = 1, day: int = 15, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, 0, tzinfo=timezone.utc)


def _event(
    eid: str,
    etype: str,
    created: datetime,
    *,
    predecessors: list[str] | None = None,
    site_id=None,
    sender=None,
    receiver=None,
    transport_km=None,
    location=None,
    input_kg=None,
    output_kg=None,
    proofs=None,
    category=None,
) -> NormalizedEvent:
    if category is None:
        category = "delivery-leg" if "delivery" in etype else "step-execution"
    return NormalizedEvent(
        event_id=eid,
        event_type=etype,
        event_category=category,
        created=created,
        site_id=site_id,
        sender_site_id=sender,
        receiver_site_id=receiver,
        transport_distance_km=transport_km,
        location=location,
        input_weight_kg=input_kg,
        output_weight_kg=output_kg,
        emission_kg_co2e=0.0,
        proofs=proofs or [],
        predecessor_ids=predecessors or [],
    )


def _ctx(
    events: list[NormalizedEvent] | None = None,
    *,
    sites: dict | None = None,
    series: dict | None = None,
    pyrolysis_window=None,
) -> NormalizedContext:
    all_cs: set[str] = set()
    for e in (events or []):
        for p in e.proofs:
            if p.file_ref and p.file_ref.cloud_storage_id and not p.file_ref.is_sensitive:
                all_cs.add(p.file_ref.cloud_storage_id)
    return NormalizedContext(
        sink_id=uuid4(),
        sink_created=_dt(2024),
        gross_impact_kg=100.0,
        net_impact_kg=80.0,
        carbon_capture_site_id=SITE_A,
        events=events or [],
        pyrolysis_window=pyrolysis_window,
        sites=sites or {},
        series=series or {},
        cloud_storage_ids_seen=all_cs,
    )


def _file_proof(cs_id: str | None = None) -> NormalizedProof:
    cs_id = cs_id or str(uuid4())
    return NormalizedProof(
        proof_id=str(uuid4()),
        timestamp=_dt(2024),
        proof_type="file",
        file_ref=FileRef(
            cloud_storage_id=cs_id,
            file_name="doc.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            is_sensitive=False,
        ),
        object_types=["material_creation"],
    )


# ---------------------------------------------------------------------------
# DAG traversal
# ---------------------------------------------------------------------------


class TestTopoLeafToRoot:
    def test_simple_chain(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1))
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"])
        e3 = _event("e3", "sink_creation", _dt(2024, 6), predecessors=["e2"])
        order = topo_leaf_to_root([e3, e1, e2])
        ids = [e.event_id for e in order]
        print(f"  → topo order: {ids}")
        assert ids.index("e1") < ids.index("e2") < ids.index("e3")

    def test_branching(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1))
        e2 = _event("e2", "biomass_creation", _dt(2024, 1))
        e3 = _event("e3", "pyrolysis", _dt(2024, 3), predecessors=["e1", "e2"])
        order = topo_leaf_to_root([e3, e1, e2])
        ids = [e.event_id for e in order]
        print(f"  → topo order (branching): {ids}")
        assert ids.index("e3") == 2


# ---------------------------------------------------------------------------
# Category 1: Removal vs. Removal
# ---------------------------------------------------------------------------


class TestTimelineOrder:
    def test_correct_order(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1))
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"])
        results = check_timeline_order(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_wrong_order(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 6))
        e2 = _event("e2", "pyrolysis", _dt(2024, 1), predecessors=["e1"])
        results = check_timeline_order(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "fail"
        assert results[0].code == "TIMELINE_ORDER"


class TestMassBalance:
    def test_balanced(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1), output_kg=1000)
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"], input_kg=1000)
        results = check_mass_balance(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_imbalanced(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1), output_kg=300)
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"], input_kg=500)
        results = check_mass_balance(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"
        assert results[0].code == "MASS_BALANCE"

    def test_no_weight_data(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1))
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"])
        results = check_mass_balance(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"


class TestSiteContinuity:
    def test_matching_sites(self):
        e1 = _event("e1", "biomass_delivery", _dt(2024, 1), receiver=SITE_A, category="delivery-leg")
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"], site_id=SITE_A)
        results = check_site_continuity(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_mismatched_sites(self):
        e1 = _event("e1", "biomass_delivery", _dt(2024, 1), receiver=SITE_A, category="delivery-leg")
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"], site_id=SITE_B)
        sites = {
            SITE_A: SiteInfo(SITE_A, "Berlin Plant", 52.52, 13.4, "Germany"),
            SITE_B: SiteInfo(SITE_B, "Munich Plant", 48.14, 11.58, "Germany"),
        }
        results = check_site_continuity(_ctx([e1, e2], sites=sites), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "fail"
        assert results[0].code == "SITE_CONTINUITY"


class TestDeliveryDistance:
    def test_plausible(self):
        e = _event(
            "e1", "biomass_delivery", _dt(2024, 1),
            sender=SITE_A, receiver=SITE_B, transport_km=600,
            category="delivery-leg",
        )
        sites = {
            SITE_A: SiteInfo(SITE_A, "Berlin", 52.52, 13.4, "DE"),
            SITE_B: SiteInfo(SITE_B, "Munich", 48.14, 11.58, "DE"),
        }
        results = check_delivery_distance(_ctx([e], sites=sites), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_implausible(self):
        e = _event(
            "e1", "biomass_delivery", _dt(2024, 1),
            sender=SITE_A, receiver=SITE_B, transport_km=50,
            category="delivery-leg",
        )
        sites = {
            SITE_A: SiteInfo(SITE_A, "Berlin", 52.52, 13.4, "DE"),
            SITE_B: SiteInfo(SITE_B, "Munich", 48.14, 11.58, "DE"),
        }
        results = check_delivery_distance(_ctx([e], sites=sites), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"
        assert results[0].code == "DELIVERY_DISTANCE"


# ---------------------------------------------------------------------------
# Category 2: Removal vs. Machine data
# ---------------------------------------------------------------------------


class TestMachineCoverage:
    def test_covered(self):
        window = (_dt(2024, 3, 1), _dt(2024, 3, 31))
        s = SeriesInfo(
            config_id=uuid4(), name="Reactor temp", unit="°C",
            machine_id=uuid4(),
            data=[(_dt(2024, 3, 15), 450.0)],
        )
        results = check_machine_coverage(
            _ctx(pyrolysis_window=window, series={s.config_id: s}), CFG
        )
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_not_covered(self):
        window = (_dt(2024, 3, 1), _dt(2024, 3, 31))
        s = SeriesInfo(
            config_id=uuid4(), name="Reactor temp", unit="°C",
            machine_id=uuid4(),
            data=[(_dt(2024, 6, 15), 450.0)],
        )
        results = check_machine_coverage(
            _ctx(pyrolysis_window=window, series={s.config_id: s}), CFG
        )
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "fail"

    def test_no_series(self):
        window = (_dt(2024, 3, 1), _dt(2024, 3, 31))
        results = check_machine_coverage(_ctx(pyrolysis_window=window), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"


class TestTempPlausible:
    def test_normal_range(self):
        s = SeriesInfo(
            config_id=uuid4(), name="Reactor temperature", unit="°C",
            machine_id=uuid4(),
            data=[(_dt(2024, 3, 15), 450.0), (_dt(2024, 3, 15, 13), 520.0)],
        )
        results = check_temp_plausible(
            _ctx(series={s.config_id: s}), CFG
        )
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_too_cold(self):
        s = SeriesInfo(
            config_id=uuid4(), name="Reactor temperature", unit="°C",
            machine_id=uuid4(),
            data=[(_dt(2024, 3, 15), 20.0), (_dt(2024, 3, 15, 13), 25.0)],
        )
        results = check_temp_plausible(
            _ctx(series={s.config_id: s}), CFG
        )
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"
        assert results[0].code == "TEMP_PLAUSIBLE"


# ---------------------------------------------------------------------------
# Category 3: Removal vs. Documents
# ---------------------------------------------------------------------------


class TestProofPresence:
    def test_proofs_present(self):
        e = _event("e1", "pyrolysis", _dt(2024, 3), proofs=[_file_proof()])
        results = check_proof_presence(_ctx([e]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_proofs_missing(self):
        e = _event("e1", "pyrolysis", _dt(2024, 3), proofs=[])
        results = check_proof_presence(_ctx([e]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "fail"
        assert results[0].code == "PROOF_PRESENCE"

    def test_sensitive_only_is_warn(self):
        sensitive_proof = NormalizedProof(
            proof_id=str(uuid4()),
            timestamp=_dt(2024),
            proof_type="file",
            file_ref=FileRef(
                cloud_storage_id=str(uuid4()),
                file_name="cert.pdf",
                mime_type="application/pdf",
                size_bytes=2048,
                is_sensitive=True,
            ),
            object_types=["pyrolysis"],
        )
        e = _event("e1", "pyrolysis", _dt(2024, 3), proofs=[sensitive_proof])
        results = check_proof_presence(_ctx([e]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"

    def test_non_critical_event_ignored(self):
        e = _event("e1", "biomass_creation", _dt(2024, 1), proofs=[])
        results = check_proof_presence(_ctx([e]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"


class TestFileReuse:
    def test_no_reuse(self):
        e1 = _event("e1", "pyrolysis", _dt(2024, 3), proofs=[_file_proof("aaa")])
        e2 = _event("e2", "sink_creation", _dt(2024, 6), proofs=[_file_proof("bbb")])
        results = check_file_reuse(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "info"

    def test_reuse_detected(self):
        shared_id = "shared-file-id"
        e1 = _event("e1", "pyrolysis", _dt(2024, 3), proofs=[_file_proof(shared_id)])
        e2 = _event("e2", "sink_creation", _dt(2024, 6), proofs=[_file_proof(shared_id)])
        results = check_file_reuse(_ctx([e1, e2]), CFG)
        print(f"  → {results[0].severity}: {results[0].message}")
        assert results[0].severity == "warn"
        assert results[0].code == "FILE_REUSE"


# ---------------------------------------------------------------------------
# Full registry runner
# ---------------------------------------------------------------------------


class TestRunRules:
    def test_runs_all_rules(self):
        e1 = _event("e1", "biomass_creation", _dt(2024, 1), output_kg=1000, proofs=[_file_proof()])
        e2 = _event("e2", "pyrolysis", _dt(2024, 3), predecessors=["e1"], input_kg=1000, proofs=[_file_proof()])
        e3 = _event("e3", "sink_creation", _dt(2024, 6), predecessors=["e2"], input_kg=300, proofs=[_file_proof()])
        window = (_dt(2024, 3, 1), _dt(2024, 3, 31))
        s = SeriesInfo(
            config_id=uuid4(), name="Reactor temp", unit="°C",
            machine_id=uuid4(),
            data=[(_dt(2024, 3, 15), 450.0)],
        )
        ctx = _ctx([e1, e2, e3], pyrolysis_window=window, series={s.config_id: s})
        results = run_rules(ctx)
        codes = {r.code for r in results}
        print(f"  → {len(results)} results, codes: {sorted(codes)}")
        for r in results:
            print(f"    [{r.severity:4s}] {r.code}: {r.message}")
        assert len(codes) == 8
