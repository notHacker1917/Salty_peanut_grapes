"""
Scoring step: reduce a list of CheckResults into a single VerificationReport.

Deterministic: start at 100, subtract per-rule weighted penalties,
clamp to 0–100, map to a confidence band, attach top reasons.

Bands:  76–100 high  |  51–75 medium  |  0–50 low

Severity semantics (set by the rule engine):
  fail = data actively contradicts the claim  → full weight penalty
  warn = data is absent / incomplete          → weight × 0.4 penalty

Rule weights reflect importance to the core question
"did real pyrolysis and carbon removal happen?"

  Tier 1 (critical — proves/disproves the removal):
    MACHINE_COVERAGE=25, TEMP_PLAUSIBLE=22, PROOF_PRESENCE=20

  Tier 2 (important — supply-chain integrity):
    TIMELINE_ORDER=15, MASS_BALANCE=15, SITE_CONTINUITY=12

  Tier 3 (supporting — suspicious but not conclusive alone):
    DELIVERY_DISTANCE=8, FILE_REUSE=5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from cula.verification.rules import CheckResult

RULE_WEIGHTS: dict[str, int] = {
    # tier 1 — critical
    "MACHINE_COVERAGE": 25,
    "TEMP_PLAUSIBLE":   22,
    "PROOF_PRESENCE":   20,
    # tier 2 — important
    "TIMELINE_ORDER":   15,
    "MASS_BALANCE":     15,
    "SITE_CONTINUITY":  12,
    # tier 3 — supporting
    "DELIVERY_DISTANCE": 8,
    "FILE_REUSE":        5,
}

DEFAULT_WEIGHT = 10
WARN_MULTIPLIER = 0.4


@dataclass
class ScoringConfig:
    rule_weights: dict[str, int] = field(default_factory=lambda: dict(RULE_WEIGHTS))
    default_weight: int = DEFAULT_WEIGHT
    warn_multiplier: float = WARN_MULTIPLIER
    band_high: int = 76
    band_medium: int = 51
    top_reasons_count: int = 3


@dataclass
class VerificationReport:
    sink_id: UUID
    checks: list[CheckResult]
    counts: dict[str, int]
    confidence_score: int          # 0–100, heuristic
    confidence_band: str           # "high" | "medium" | "low"
    top_reasons: list[str]


def _penalty(check: CheckResult, cfg: ScoringConfig) -> float:
    weight = cfg.rule_weights.get(check.code, cfg.default_weight)
    if check.severity == "fail":
        return float(weight)
    if check.severity == "warn":
        return weight * cfg.warn_multiplier
    return 0.0


def score(
    sink_id: UUID,
    checks: list[CheckResult],
    cfg: ScoringConfig | None = None,
) -> VerificationReport:
    """Aggregate CheckResults into a VerificationReport."""
    cfg = cfg or ScoringConfig()

    counts = {"fail": 0, "warn": 0, "info": 0}
    for c in checks:
        counts[c.severity] = counts.get(c.severity, 0) + 1

    total_penalty = sum(_penalty(c, cfg) for c in checks)
    points = max(0, min(100, round(100 - total_penalty)))

    if points >= cfg.band_high:
        band = "high"
    elif points >= cfg.band_medium:
        band = "medium"
    else:
        band = "low"

    non_info = [c for c in checks if c.severity != "info"]
    non_info.sort(key=lambda c: _penalty(c, cfg), reverse=True)
    top_reasons = [c.message for c in non_info[: cfg.top_reasons_count]]

    return VerificationReport(
        sink_id=sink_id,
        checks=checks,
        counts=counts,
        confidence_score=points,
        confidence_band=band,
        top_reasons=top_reasons,
    )
