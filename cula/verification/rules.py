"""
Rule engine: deterministic cross-checks over a NormalizedContext.

Three categories:
  1. Removal vs. Removal  (internal consistency — DAG walk, leaf → root)
  2. Removal vs. Machine data
  3. Removal vs. Documents

Each rule is a pure function  (ctx, cfg) -> list[CheckResult].
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from cula.verification.normalize import NormalizedContext, NormalizedEvent


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    code: str
    severity: str       # "info" | "warn" | "fail"
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleConfig:
    mass_balance_tolerance: float = 0.15
    distance_ratio_min: float = 0.5
    distance_ratio_max: float = 2.5
    temp_min_c: float = 200.0
    temp_max_c: float = 1200.0
    critical_event_types: tuple[str, ...] = ("pyrolysis", "sink_creation")


RuleFn = Callable[[NormalizedContext, RuleConfig], list[CheckResult]]


# ---------------------------------------------------------------------------
# DAG traversal — Kahn's algorithm, leaf → root
# ---------------------------------------------------------------------------


def topo_leaf_to_root(events: list[NormalizedEvent]) -> list[NormalizedEvent]:
    """Topological order starting from leaves (no predecessors) toward root."""
    by_id: dict[str, NormalizedEvent] = {e.event_id: e for e in events}
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {e.event_id: 0 for e in events}

    for e in events:
        for pred_id in e.predecessor_ids:
            if pred_id in by_id:
                children[pred_id].append(e.event_id)
                in_degree[e.event_id] += 1

    queue: deque[str] = deque(
        eid for eid, deg in in_degree.items() if deg == 0
    )
    order: list[NormalizedEvent] = []
    while queue:
        eid = queue.popleft()
        order.append(by_id[eid])
        for child_id in children[eid]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    return order


def _iter_edges(
    events: list[NormalizedEvent],
) -> list[tuple[NormalizedEvent, NormalizedEvent]]:
    """Yield (predecessor, successor) pairs from predecessor_ids."""
    by_id = {e.event_id: e for e in events}
    edges: list[tuple[NormalizedEvent, NormalizedEvent]] = []
    for e in events:
        for pred_id in e.predecessor_ids:
            pred = by_id.get(pred_id)
            if pred is not None:
                edges.append((pred, e))
    return edges


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Category 1: Removal vs. Removal (internal consistency)
# ---------------------------------------------------------------------------


def check_timeline_order(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Successor created must be >= predecessor created along each edge."""
    results: list[CheckResult] = []
    for pred, succ in _iter_edges(ctx.events):
        if succ.created < pred.created:
            results.append(CheckResult(
                code="TIMELINE_ORDER",
                severity="fail",
                message=(
                    f"{succ.event_type} ({succ.created:%Y-%m-%d %H:%M}) "
                    f"is before its predecessor {pred.event_type} "
                    f"({pred.created:%Y-%m-%d %H:%M})"
                ),
                evidence={
                    "predecessor": pred.event_id,
                    "successor": succ.event_id,
                    "pred_created": pred.created.isoformat(),
                    "succ_created": succ.created.isoformat(),
                },
            ))
    if not results:
        results.append(CheckResult(
            code="TIMELINE_ORDER", severity="info",
            message="All event timestamps follow chronological order.",
        ))
    return results


def check_mass_balance(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Output weight of predecessor ≈ input weight of successor (within tolerance)."""
    results: list[CheckResult] = []
    for pred, succ in _iter_edges(ctx.events):
        out_kg = pred.output_weight_kg
        in_kg = succ.input_weight_kg
        if out_kg is None or in_kg is None:
            continue
        if out_kg == 0:
            continue
        ratio = abs(in_kg - out_kg) / out_kg
        if ratio > cfg.mass_balance_tolerance:
            results.append(CheckResult(
                code="MASS_BALANCE",
                severity="warn",
                message=(
                    f"Weight mismatch: {pred.event_type} output={out_kg:.1f}kg "
                    f"→ {succ.event_type} input={in_kg:.1f}kg "
                    f"(diff {ratio:.0%}, tolerance {cfg.mass_balance_tolerance:.0%})"
                ),
                evidence={
                    "predecessor": pred.event_id,
                    "successor": succ.event_id,
                    "output_kg": out_kg,
                    "input_kg": in_kg,
                    "ratio": round(ratio, 3),
                },
            ))
    if not results:
        results.append(CheckResult(
            code="MASS_BALANCE", severity="info",
            message="Mass balance consistent across all edges with weight data.",
        ))
    return results


def check_site_continuity(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Delivery receiver_site_id must equal the next processing step's site_id."""
    results: list[CheckResult] = []
    for pred, succ in _iter_edges(ctx.events):
        if pred.event_category != "delivery-leg":
            continue
        receiver = pred.receiver_site_id
        next_site = succ.site_id
        if receiver is None or next_site is None:
            continue
        if receiver != next_site:
            recv_name = ctx.sites[receiver].name if receiver in ctx.sites else str(receiver)
            next_name = ctx.sites[next_site].name if next_site in ctx.sites else str(next_site)
            results.append(CheckResult(
                code="SITE_CONTINUITY",
                severity="fail",
                message=(
                    f"Delivery to {recv_name} but {succ.event_type} "
                    f"happened at {next_name}"
                ),
                evidence={
                    "delivery_event": pred.event_id,
                    "successor_event": succ.event_id,
                    "receiver_site": str(receiver),
                    "successor_site": str(next_site),
                },
            ))
    if not results:
        results.append(CheckResult(
            code="SITE_CONTINUITY", severity="info",
            message="Delivery destinations match subsequent processing sites.",
        ))
    return results


def check_delivery_distance(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Claimed transport km vs haversine between sender and receiver sites."""
    results: list[CheckResult] = []
    for e in ctx.events:
        if e.event_category != "delivery-leg":
            continue
        if e.transport_distance_km is None:
            continue
        sender = ctx.sites.get(e.sender_site_id) if e.sender_site_id else None
        receiver = ctx.sites.get(e.receiver_site_id) if e.receiver_site_id else None
        if sender is None or receiver is None:
            continue
        crow = _haversine_km(sender.lat, sender.lon, receiver.lat, receiver.lon)
        if crow < 1.0:
            continue
        ratio = e.transport_distance_km / crow
        if ratio < cfg.distance_ratio_min or ratio > cfg.distance_ratio_max:
            results.append(CheckResult(
                code="DELIVERY_DISTANCE",
                severity="warn",
                message=(
                    f"{e.event_type}: claimed {e.transport_distance_km:.0f}km "
                    f"but straight-line {sender.name}→{receiver.name} is {crow:.0f}km "
                    f"(ratio {ratio:.1f}x)"
                ),
                evidence={
                    "event_id": e.event_id,
                    "claimed_km": e.transport_distance_km,
                    "haversine_km": round(crow, 1),
                    "ratio": round(ratio, 2),
                },
            ))
    if not results:
        results.append(CheckResult(
            code="DELIVERY_DISTANCE", severity="info",
            message="Delivery distances are plausible relative to site locations.",
        ))
    return results


# ---------------------------------------------------------------------------
# Category 2: Removal vs. Machine data
# ---------------------------------------------------------------------------


def check_machine_coverage(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """At least one series must have data points during the pyrolysis window."""
    if not ctx.pyrolysis_window:
        return [CheckResult(
            code="MACHINE_COVERAGE", severity="warn",
            message="No pyrolysis window — cannot check machine coverage.",
        )]

    if not ctx.series:
        return [CheckResult(
            code="MACHINE_COVERAGE", severity="warn",
            message="No machine series data available to corroborate pyrolysis.",
        )]

    win_start, win_end = ctx.pyrolysis_window
    covered = False
    for s in ctx.series.values():
        for ts, _ in s.data:
            if win_start <= ts <= win_end:
                covered = True
                break
        if covered:
            break

    if not covered:
        return [CheckResult(
            code="MACHINE_COVERAGE", severity="fail",
            message=(
                f"No series has data points within the pyrolysis window "
                f"({win_start:%Y-%m-%d %H:%M} – {win_end:%Y-%m-%d %H:%M})."
            ),
            evidence={
                "window_start": win_start.isoformat(),
                "window_end": win_end.isoformat(),
                "series_count": len(ctx.series),
            },
        )]

    return [CheckResult(
        code="MACHINE_COVERAGE", severity="info",
        message="Machine series data present during pyrolysis window.",
    )]


def check_temp_plausible(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Temperature-like series values should be in a plausible range during pyrolysis."""
    results: list[CheckResult] = []
    temp_keywords = ("temp", "temperatur", "reactor", "°c")

    for s in ctx.series.values():
        name_lower = s.name.lower()
        unit_lower = (s.unit or "").lower()
        is_temp = any(kw in name_lower or kw in unit_lower for kw in temp_keywords)
        if not is_temp:
            continue

        if not s.data:
            continue

        values = [v for _, v in s.data]
        lo, hi = min(values), max(values)

        if hi < cfg.temp_min_c:
            results.append(CheckResult(
                code="TEMP_PLAUSIBLE",
                severity="warn",
                message=(
                    f"Series '{s.name}' max value is {hi:.1f}°C — "
                    f"below minimum expected {cfg.temp_min_c:.0f}°C for pyrolysis."
                ),
                evidence={"series": s.name, "min": lo, "max": hi},
            ))
        elif lo > cfg.temp_max_c:
            results.append(CheckResult(
                code="TEMP_PLAUSIBLE",
                severity="warn",
                message=(
                    f"Series '{s.name}' min value is {lo:.1f}°C — "
                    f"above maximum expected {cfg.temp_max_c:.0f}°C."
                ),
                evidence={"series": s.name, "min": lo, "max": hi},
            ))

    if not results:
        results.append(CheckResult(
            code="TEMP_PLAUSIBLE", severity="info",
            message="Temperature series values are within plausible range.",
        ))
    return results


# ---------------------------------------------------------------------------
# Category 3: Removal vs. Documents
# ---------------------------------------------------------------------------


def check_proof_presence(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Critical event types must have at least one file proof.

    Sensitive proofs count — the proof exists, it's just access-restricted.
    fail  = zero proofs of any kind on a critical event.
    warn  = proofs exist but all are sensitive (can't independently verify).
    """
    no_proof_by_type: dict[str, list[str]] = defaultdict(list)
    sensitive_only_by_type: dict[str, list[str]] = defaultdict(list)

    for e in ctx.events:
        if e.event_type not in cfg.critical_event_types:
            continue
        file_proofs = [
            p for p in e.proofs
            if p.proof_type == "file" and p.file_ref is not None
        ]
        if not file_proofs:
            no_proof_by_type[e.event_type].append(e.event_id)
        elif all(p.file_ref.is_sensitive for p in file_proofs):
            sensitive_only_by_type[e.event_type].append(e.event_id)

    if not no_proof_by_type and not sensitive_only_by_type:
        return [CheckResult(
            code="PROOF_PRESENCE", severity="info",
            message="All critical events have file proofs.",
        )]

    results: list[CheckResult] = []
    for etype, event_ids in no_proof_by_type.items():
        n = len(event_ids)
        noun = "event has" if n == 1 else "events have"
        results.append(CheckResult(
            code="PROOF_PRESENCE",
            severity="fail",
            message=f"{n} {etype} {noun} no file proof at all.",
            evidence={"event_type": etype, "event_ids": event_ids},
        ))
    for etype, event_ids in sensitive_only_by_type.items():
        n = len(event_ids)
        noun = "event has" if n == 1 else "events have"
        results.append(CheckResult(
            code="PROOF_PRESENCE",
            severity="warn",
            message=f"{n} {etype} {noun} only sensitive (non-downloadable) proofs.",
            evidence={"event_type": etype, "event_ids": event_ids},
        ))
    return results


def check_file_reuse(
    ctx: NormalizedContext, cfg: RuleConfig,
) -> list[CheckResult]:
    """Same cloudStorageId on proofs across unrelated event types."""
    results: list[CheckResult] = []
    id_to_event_types: dict[str, set[str]] = defaultdict(set)

    for e in ctx.events:
        for p in e.proofs:
            if p.file_ref and p.file_ref.cloud_storage_id and not p.file_ref.is_sensitive:
                id_to_event_types[p.file_ref.cloud_storage_id].add(e.event_type)

    for cs_id, event_types in id_to_event_types.items():
        if len(event_types) > 1:
            results.append(CheckResult(
                code="FILE_REUSE",
                severity="warn",
                message=(
                    f"File {cs_id[:8]}… is used as proof for "
                    f"{len(event_types)} different event types: "
                    f"{', '.join(sorted(event_types))}"
                ),
                evidence={
                    "cloud_storage_id": cs_id,
                    "event_types": sorted(event_types),
                },
            ))
    if not results:
        results.append(CheckResult(
            code="FILE_REUSE", severity="info",
            message="No proof files are reused across different event types.",
        ))
    return results


# ---------------------------------------------------------------------------
# Registry and runner
# ---------------------------------------------------------------------------

RULE_REGISTRY: list[RuleFn] = [
    # Category 1: Removal vs. Removal
    check_timeline_order,
    check_mass_balance,
    check_site_continuity,
    check_delivery_distance,
    # Category 2: Removal vs. Machine data
    check_machine_coverage,
    check_temp_plausible,
    # Category 3: Removal vs. Documents
    check_proof_presence,
    check_file_reuse,
]


def run_rules(
    ctx: NormalizedContext,
    cfg: RuleConfig | None = None,
    rules: list[RuleFn] | None = None,
) -> list[CheckResult]:
    """Execute all rules (or a custom subset) and return combined results."""
    cfg = cfg or RuleConfig()
    registry = rules if rules is not None else RULE_REGISTRY
    results: list[CheckResult] = []
    for rule_fn in registry:
        results.extend(rule_fn(ctx, cfg))
    return results
