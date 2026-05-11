#!/usr/bin/env python3
"""
Scan ALL sinks: run the verification pipeline on each, then cross-compare
to surface injected fake data.

Steps:
  1. Fetch + normalize + run rules on every sink
  2. Cross-sink comparison (shared proof files, shared machines)
  3. Statistical outliers on machine series (flatline, too few unique values)
  4. Timestamp clustering (all events suspiciously close together)

Usage:
    python example/find_fakes.py
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from datetime import timedelta
from uuid import UUID

from cula import CulaClient
from cula.verification.fetch import fetch_sink_data
from cula.verification.normalize import NormalizedContext, normalize
from cula.verification.rules import CheckResult, RuleConfig, run_rules
from cula.verification.scoring import VerificationReport, score


def _sep(label: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {label}")
    print(f"{'═' * 64}")


def main() -> None:
    cfg = RuleConfig()

    # ------------------------------------------------------------------
    # Step 1: fetch + normalize + rules for ALL sinks
    # ------------------------------------------------------------------
    _sep("STEP 1 — Fetch, normalize, and run rules on all sinks")

    with CulaClient() as client:
        sink_ids = client.list_sinks()
        print(f"  Found {len(sink_ids)} sinks\n")

        results: dict[UUID, dict] = {}
        for i, sid in enumerate(sink_ids):
            print(f"  [{i+1}/{len(sink_ids)}] {sid} …", end="", flush=True)
            fetch = fetch_sink_data(client, sid)
            ctx = normalize(fetch)
            checks = run_rules(ctx, cfg)
            report = score(sid, checks)
            results[sid] = {"ctx": ctx, "report": report, "fetch_errors": fetch.errors}
            print(f"  score={report.confidence_score:3d} ({report.confidence_band})")
            time.sleep(0.2)

    # rank by score ascending (lowest confidence first)
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1]["report"].confidence_score,
    )

    print("\n  Top suspects (lowest confidence):")
    for sid, r in ranked[:10]:
        rpt: VerificationReport = r["report"]
        print(f"    {sid}  score={rpt.confidence_score} ({rpt.confidence_band})")
        for reason in rpt.top_reasons:
            print(f"      → {reason}")

    # ------------------------------------------------------------------
    # Step 2: Cross-sink comparison
    # ------------------------------------------------------------------
    _sep("STEP 2 — Cross-sink comparison")

    # 2a: shared proof files across sinks
    cs_id_to_sinks: dict[str, set[UUID]] = defaultdict(set)
    for sid, r in results.items():
        ctx: NormalizedContext = r["ctx"]
        for cs_id in ctx.cloud_storage_ids_seen:
            cs_id_to_sinks[cs_id].add(sid)

    shared_proofs = {cs_id: sinks for cs_id, sinks in cs_id_to_sinks.items() if len(sinks) > 1}
    if shared_proofs:
        print(f"  {len(shared_proofs)} proof file(s) shared across sinks:")
        for cs_id, sinks in shared_proofs.items():
            print(f"    file {cs_id[:12]}… → {len(sinks)} sinks: {[str(s)[:8] for s in sinks]}")
    else:
        print("  No proof files shared across sinks.")

    # 2b: sinks referencing same capture site
    site_to_sinks: dict[UUID, list[UUID]] = defaultdict(list)
    for sid, r in results.items():
        ctx = r["ctx"]
        if ctx.carbon_capture_site_id:
            site_to_sinks[ctx.carbon_capture_site_id].append(sid)

    multi_site = {s: sids for s, sids in site_to_sinks.items() if len(sids) > 1}
    if multi_site:
        print(f"\n  {len(multi_site)} capture site(s) used by multiple sinks:")
        for site_id, sids in multi_site.items():
            print(f"    site {str(site_id)[:12]}… → {len(sids)} sinks")
    else:
        print("  Each sink uses a unique capture site.")

    # ------------------------------------------------------------------
    # Step 3: Statistical outliers on machine series
    # ------------------------------------------------------------------
    _sep("STEP 3 — Series anomalies (flatline, low variance, too few values)")

    for sid, r in results.items():
        ctx = r["ctx"]
        for s in ctx.series.values():
            if len(s.data) < 3:
                continue
            values = [v for _, v in s.data]
            unique = len(set(values))
            sd = statistics.stdev(values)

            flags: list[str] = []
            if sd < 0.01:
                flags.append(f"flatline (stdev={sd:.4f})")
            if unique == 1:
                flags.append(f"single repeated value ({values[0]:.2f})")
            elif unique <= 3 and len(values) > 10:
                flags.append(f"only {unique} unique values in {len(values)} points")

            if flags:
                print(f"  sink {str(sid)[:8]}…  series '{s.name}': {', '.join(flags)}")

    # ------------------------------------------------------------------
    # Step 4: Timestamp clustering
    # ------------------------------------------------------------------
    _sep("STEP 4 — Timestamp clustering (events too close together)")

    CLUSTER_THRESHOLD = timedelta(minutes=10)

    for sid, r in results.items():
        ctx = r["ctx"]
        if len(ctx.events) < 3:
            continue
        timestamps = sorted(e.created for e in ctx.events)
        span = timestamps[-1] - timestamps[0]
        if span < CLUSTER_THRESHOLD:
            print(
                f"  sink {str(sid)[:8]}…  "
                f"{len(ctx.events)} events within {span} "
                f"({timestamps[0]:%Y-%m-%d %H:%M} – {timestamps[-1]:%H:%M})"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _sep("SUMMARY — Likely fakes")

    suspect_sinks: dict[UUID, list[str]] = defaultdict(list)

    for sid, r in results.items():
        rpt: VerificationReport = r["report"]
        if rpt.confidence_band in ("low", "medium"):
            suspect_sinks[sid].append(f"score={rpt.confidence_score} ({rpt.confidence_band})")

    for cs_id, sinks in shared_proofs.items():
        for sid in sinks:
            suspect_sinks[sid].append(f"shares proof file {cs_id[:8]}…")

    for sid, r in results.items():
        ctx = r["ctx"]
        if len(ctx.events) >= 3:
            timestamps = sorted(e.created for e in ctx.events)
            if timestamps[-1] - timestamps[0] < CLUSTER_THRESHOLD:
                suspect_sinks[sid].append("timestamp cluster")

    for sid, r in results.items():
        ctx = r["ctx"]
        for s in ctx.series.values():
            if len(s.data) >= 3:
                values = [v for _, v in s.data]
                if statistics.stdev(values) < 0.01:
                    suspect_sinks[sid].append(f"flatline series '{s.name}'")
                    break

    if suspect_sinks:
        ranked_suspects = sorted(suspect_sinks.items(), key=lambda kv: len(kv[1]), reverse=True)
        for sid, reasons in ranked_suspects:
            print(f"  {sid}")
            for r in reasons:
                print(f"    → {r}")
    else:
        print("  No strong suspects found.")

    print()


if __name__ == "__main__":
    main()
