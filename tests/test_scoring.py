"""Tests for cula.verification.scoring (weighted, per-rule penalties)."""

from __future__ import annotations

from uuid import uuid4

from cula.verification.rules import CheckResult
from cula.verification.scoring import RULE_WEIGHTS, ScoringConfig, score

SINK_ID = uuid4()


def _check(severity: str, code: str = "TEST", msg: str = "test") -> CheckResult:
    return CheckResult(code=code, severity=severity, message=msg)


class TestBands:
    """Verify the 0-50 low / 51-75 medium / 76-100 high thresholds."""

    def test_all_info_is_high(self):
        checks = [_check("info"), _check("info")]
        report = score(SINK_ID, checks)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score == 100
        assert report.confidence_band == "high"
        assert report.top_reasons == []

    def test_76_is_high(self):
        """A single PROOF_PRESENCE warn (20*0.4=8) → 92, still high."""
        checks = [_check("warn", code="PROOF_PRESENCE", msg="minor")]
        report = score(SINK_ID, checks)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score >= 76
        assert report.confidence_band == "high"

    def test_medium_band(self):
        """MACHINE_COVERAGE fail (25) → 75 → medium."""
        checks = [_check("fail", code="MACHINE_COVERAGE", msg="no data")]
        report = score(SINK_ID, checks)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score == 75
        assert report.confidence_band == "medium"

    def test_low_band(self):
        """MACHINE_COVERAGE + TEMP_PLAUSIBLE fail (25+22=47) → 53 → medium,
        add PROOF_PRESENCE fail (20) → 33 → low."""
        checks = [
            _check("fail", code="MACHINE_COVERAGE", msg="no machine data"),
            _check("fail", code="TEMP_PLAUSIBLE", msg="too cold"),
            _check("fail", code="PROOF_PRESENCE", msg="missing proofs"),
        ]
        report = score(SINK_ID, checks)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score == 33
        assert report.confidence_band == "low"


class TestWeightedPenalties:
    """Each rule code has a different impact."""

    def test_tier1_machine_coverage_fail(self):
        checks = [_check("fail", code="MACHINE_COVERAGE", msg="no data")]
        report = score(SINK_ID, checks)
        print(f"  → MACHINE_COVERAGE fail: score={report.confidence_score}")
        assert report.confidence_score == 75

    def test_tier1_temp_plausible_fail(self):
        checks = [_check("fail", code="TEMP_PLAUSIBLE", msg="cold")]
        report = score(SINK_ID, checks)
        print(f"  → TEMP_PLAUSIBLE fail: score={report.confidence_score}")
        assert report.confidence_score == 78

    def test_tier3_file_reuse_fail(self):
        checks = [_check("fail", code="FILE_REUSE", msg="reused")]
        report = score(SINK_ID, checks)
        print(f"  → FILE_REUSE fail: score={report.confidence_score}")
        assert report.confidence_score == 95

    def test_warn_uses_multiplier(self):
        """MACHINE_COVERAGE warn: 25 * 0.4 = 10 → score 90."""
        checks = [_check("warn", code="MACHINE_COVERAGE", msg="partial")]
        report = score(SINK_ID, checks)
        print(f"  → MACHINE_COVERAGE warn: score={report.confidence_score}")
        assert report.confidence_score == 90

    def test_unknown_code_gets_default_weight(self):
        checks = [_check("fail", code="UNKNOWN_RULE", msg="mystery")]
        report = score(SINK_ID, checks)
        print(f"  → UNKNOWN_RULE fail: score={report.confidence_score}")
        assert report.confidence_score == 90

    def test_clamps_at_zero(self):
        checks = [_check("fail", code=code) for code in RULE_WEIGHTS] * 3
        report = score(SINK_ID, checks)
        print(f"  → score={report.confidence_score}")
        assert report.confidence_score == 0


class TestTopReasons:
    """Top reasons are ordered by penalty weight (highest first)."""

    def test_ordered_by_weight(self):
        checks = [
            _check("fail", code="FILE_REUSE", msg="reused file"),
            _check("fail", code="MACHINE_COVERAGE", msg="no machine data"),
            _check("warn", code="TEMP_PLAUSIBLE", msg="temps low"),
        ]
        report = score(SINK_ID, checks)
        print(f"  → top_reasons={report.top_reasons}")
        assert report.top_reasons[0] == "no machine data"
        assert report.top_reasons[1] == "temps low"
        assert report.top_reasons[2] == "reused file"

    def test_capped_at_three(self):
        checks = [_check("fail", code=code, msg=code) for code in RULE_WEIGHTS]
        report = score(SINK_ID, checks)
        print(f"  → top_reasons ({len(report.top_reasons)}): {report.top_reasons}")
        assert len(report.top_reasons) == 3

    def test_info_excluded(self):
        checks = [
            _check("info", code="TIMELINE_ORDER", msg="all good"),
            _check("fail", code="MASS_BALANCE", msg="bad weight"),
        ]
        report = score(SINK_ID, checks)
        print(f"  → top_reasons={report.top_reasons}")
        assert report.top_reasons == ["bad weight"]


class TestCustomConfig:
    def test_override_weight(self):
        cfg = ScoringConfig(rule_weights={"FILE_REUSE": 50})
        checks = [_check("fail", code="FILE_REUSE", msg="big hit")]
        report = score(SINK_ID, checks, cfg)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score == 50
        assert report.confidence_band == "low"

    def test_override_bands(self):
        cfg = ScoringConfig(band_high=90, band_medium=60)
        checks = [_check("fail", code="DELIVERY_DISTANCE", msg="far")]
        report = score(SINK_ID, checks, cfg)
        print(f"  → score={report.confidence_score} band={report.confidence_band}")
        assert report.confidence_score == 92
        assert report.confidence_band == "high"

    def test_report_identity(self):
        checks = [_check("info"), _check("warn", code="MASS_BALANCE"), _check("fail", code="PROOF_PRESENCE")]
        report = score(SINK_ID, checks)
        print(f"  → checks in report: {len(report.checks)}")
        assert report.checks is checks
        assert report.sink_id == SINK_ID
