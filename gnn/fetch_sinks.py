#!/usr/bin/env python3
"""
Download every sink from the Cula API and save a local NetworkX node-link JSON
per sink (same shape as :func:`networkx.node_link_data` / ``sink.json``).

Requires the ``cula`` package (repo root on ``PYTHONPATH`` or ``pip install -e .``).

Environment:

- ``CULA_API_BASE_URL`` — optional override for the API host (default: production hack-hpi URL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from pydantic import ValidationError
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx  # noqa: E402

from cula.client import DEFAULT_BASE_URL, CulaClient  # noqa: E402
from cula.sink_graph import build_entity_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default=None,
        help=f"API base URL (default: env CULA_API_BASE_URL or {DEFAULT_BASE_URL!r})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sinks",
        help="Directory to write <sink-uuid>.json files",
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional path to write sinks_index.json (manifest of ids and labels)",
    )
    p.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Path for sinks_fetch_failures.json (skipped sinks and error details)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_url = (
        args.base_url or os.environ.get("CULA_API_BASE_URL") or DEFAULT_BASE_URL
    )
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []

    with CulaClient(base_url=base_url) as client:
        sink_ids = client.list_sinks()
        total = len(sink_ids)
        for i, sid in enumerate(sink_ids, start=1):
            try:
                sink = client.get_sink(sid)
                graph = build_entity_graph(sink)
            except ValidationError as exc:
                failures.append(
                    {
                        "id": str(sid),
                        "error": "validation_error",
                        "detail": exc.errors(),
                    }
                )
                print(
                    f"[{i}/{total}] SKIP {sid} (Pydantic validation); see failures log",
                    flush=True,
                )
                continue
            except Exception as exc:
                failures.append(
                    {
                        "id": str(sid),
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"[{i}/{total}] SKIP {sid} ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                continue

            payload = nx.node_link_data(graph)
            path = out_dir / f"{sid}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            label = ""
            for n, data in graph.nodes(data=True):
                if data.get("kind") == "sink":
                    label = str(data.get("label", ""))
                    break
            manifest.append({"id": str(sid), "path": str(path), "label": label})
            print(f"[{i}/{total}] wrote {path.name} ({graph.number_of_nodes()} nodes)", flush=True)

    index_path = args.index or (out_dir.parent / "sinks_index.json")
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {index_path} ({len(manifest)} sinks)", flush=True)

    if failures:
        fail_path = args.failures or (out_dir.parent / "sinks_fetch_failures.json")
        fail_path.write_text(
            json.dumps(failures, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            f"Wrote {len(failures)} failure(s) to {fail_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
