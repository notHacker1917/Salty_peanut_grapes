#!/usr/bin/env python3
"""
Run the full verification pipeline (fetch → normalize → rules) against
the live Cula API and print a diagnostic summary.

Usage:
    python live_demo.py                 # first sink
    python live_demo.py <UUID>          # specific sink
    python live_demo.py --all           # all sinks (overview)
    python live_demo.py --all --limit 5 # first 5 sinks only
"""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from cula import CulaClient
from cula.verification.fetch import fetch_sink_data
from cula.verification.normalize import normalize
from cula.verification.rules import RuleConfig, run_rules
from cula.verification.scoring import score


def _print_separator(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def inspect_one(client: CulaClient, sink_id: UUID) -> None:
    """Fetch, normalize, and print a detailed breakdown for one sink."""

    print(f"Fetching sink {sink_id} …")
    result = fetch_sink_data(client, sink_id)
    ctx = normalize(result)

    _print_separator("SINK")
    print(f"  id:              {ctx.sink_id}")
    print(f"  created:         {ctx.sink_created}")
    print(f"  gross impact:    {ctx.gross_impact_kg} kg CO₂e")
    print(f"  net impact:      {ctx.net_impact_kg} kg CO₂e")
    print(f"  capture site:    {ctx.carbon_capture_site_id}")

    _print_separator(f"EVENTS ({len(ctx.events)})")
    for e in ctx.events:
        proofs_summary = f"{len(e.proofs)} proofs"
        file_proofs = sum(1 for p in e.proofs if p.file_ref and not p.file_ref.is_sensitive)
        if file_proofs:
            proofs_summary += f" ({file_proofs} files)"
        weight = ""
        if e.input_weight_kg is not None:
            weight += f"  in={e.input_weight_kg:.1f}kg"
        if e.output_weight_kg is not None:
            weight += f"  out={e.output_weight_kg:.1f}kg"
        print(
            f"  {e.event_type:35s}  {e.created:%Y-%m-%d %H:%M}  "
            f"emission={e.emission_kg_co2e:>8.2f}  {proofs_summary}{weight}"
        )
        if e.predecessor_ids:
            print(f"    └─ predecessors: {e.predecessor_ids}")

    _print_separator("PYROLYSIS WINDOW")
    if ctx.pyrolysis_window:
        start, end = ctx.pyrolysis_window
        print(f"  {start}  →  {end}")
    else:
        print("  (none)")

    _print_separator(f"SITES ({len(ctx.sites)})")
    for sid, s in ctx.sites.items():
        print(f"  {s.name:30s}  ({s.lat:.4f}, {s.lon:.4f})  {s.country or '?'}")

    _print_separator(f"MACHINE SERIES ({len(ctx.series)})")
    for sid, s in ctx.series.items():
        print(f"  {s.name:30s}  unit={s.unit or '?':6s}  points={len(s.data)}")
        if s.data:
            values = [v for _, v in s.data]
            print(f"    range: {min(values):.2f} – {max(values):.2f}   "
                  f"first={s.data[0][0]:%Y-%m-%d %H:%M}  last={s.data[-1][0]:%Y-%m-%d %H:%M}")

    _print_separator("PROOF FILES")
    print(f"  unique cloudStorageIds seen: {len(ctx.cloud_storage_ids_seen)}")

    # --- rule engine -------------------------------------------------------
    checks = run_rules(ctx, RuleConfig())
    fails = [c for c in checks if c.severity == "fail"]
    warns = [c for c in checks if c.severity == "warn"]
    infos = [c for c in checks if c.severity == "info"]

    _print_separator(f"RULE RESULTS ({len(fails)} fail, {len(warns)} warn, {len(infos)} info)")
    for c in checks:
        icon = {"fail": "✗", "warn": "⚠", "info": "✓"}.get(c.severity, "?")
        print(f"  {icon} [{c.severity:4s}] {c.code:22s}  {c.message}")

    # --- scoring -----------------------------------------------------------
    report = score(sink_id, checks)
    _print_separator("CONFIDENCE SCORE")
    print(f"  score:  {report.confidence_score}/100  ({report.confidence_band})")
    print(f"  counts: {report.counts}")
    if report.top_reasons:
        print(f"  top reasons:")
        for r in report.top_reasons:
            print(f"    → {r}")

    if result.errors:
        _print_separator(f"FETCH ERRORS ({len(result.errors)})")
        for err in result.errors:
            print(f"  ⚠ {err}")

    print()


def overview(client: CulaClient, sink_ids: list[UUID], limit: int | None = None) -> None:
    """Print a one-line summary per sink."""
    sink_ids = sink_ids[:limit] if limit else sink_ids
    _print_separator(f"OVERVIEW ({len(sink_ids)} sinks)")
    cfg = RuleConfig()
    for sid in sink_ids:
        result = fetch_sink_data(client, sid)
        ctx = normalize(result)
        checks = run_rules(ctx, cfg)
        report = score(sid, checks)
        errs = f"  fetch_errors={len(result.errors)}" if result.errors else ""
        print(
            f"  {sid}  events={len(ctx.events):2d}  "
            f"series={len(ctx.series):2d}  "
            f"score={report.confidence_score:3d} ({report.confidence_band:6s})  "
            f"fail={report.counts['fail']} warn={report.counts['warn']}  "
            f"net={ctx.net_impact_kg or 0:.1f}kg{errs}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full verification pipeline on live sinks and print diagnostics."
    )
    parser.add_argument(
        "sink_id", nargs="?", metavar="UUID",
        help="Sink id (default: first listed sink)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Print a one-line overview for all sinks",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="With --all: only show the first N sinks",
    )
    args = parser.parse_args()

    with CulaClient() as client:
        ids = client.list_sinks()
        if not ids:
            print("No sinks returned by the API.", file=sys.stderr)
            return 1

        if args.all:
            overview(client, ids, limit=args.limit)
            return 0

        if args.sink_id:
            try:
                sink_id = UUID(args.sink_id)
            except ValueError:
                print(f"Invalid UUID: {args.sink_id!r}", file=sys.stderr)
                return 1
        else:
            sink_id = ids[0]

        inspect_one(client, sink_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
