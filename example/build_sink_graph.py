#!/usr/bin/env python3
"""
Fetch a sink from the Cula API and build a NetworkX :class:`~networkx.DiGraph`.

Prints a short summary to stdout. With ``-o``, writes the graph to a file using
GraphML, GEXF, or JSON (node-link format).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

import networkx as nx

from cula import CulaClient
from cula.sink_graph import build_entity_graph, graph_summary, sink_title


def _graph_for_xml_export(G: nx.DiGraph) -> nx.DiGraph:
    """
    GraphML/GEXF only accept scalar attribute values. ``registry`` (dict) and
    ``aggregated_links`` (list of dicts) are JSON-stringified.
    """
    H = G.copy()
    for _n, d in H.nodes(data=True):
        reg = d.get("registry")
        if isinstance(reg, dict):
            d["registry_json"] = json.dumps(reg, ensure_ascii=False)
            del d["registry"]
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, (dict, list)):
                d[k] = json.dumps(v, ensure_ascii=False)
    for _u, _v, d in H.edges(data=True):
        links = d.get("aggregated_links")
        if isinstance(links, list):
            d["aggregated_links_json"] = json.dumps(links, ensure_ascii=False)
            del d["aggregated_links"]
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, (dict, list)):
                d[k] = json.dumps(v, ensure_ascii=False)
    return H


def _write_graph(G: nx.DiGraph, path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "json":
        data = nx.node_link_data(G)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return
    if fmt == "graphml":
        nx.write_graphml(_graph_for_xml_export(G), path)
        return
    if fmt == "gexf":
        nx.write_gexf(_graph_for_xml_export(G), path)
        return
    raise ValueError(f"Unknown format: {fmt!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a NetworkX graph for a carbon sink and print stats (optionally export)."
    )
    parser.add_argument(
        "sink_id",
        nargs="?",
        metavar="UUID",
        help="Sink id (default: first id from list_sinks)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=("graphml", "gexf", "json"),
        default="json",
        help="File format when using -o (default: json)",
    )
    args = parser.parse_args()

    with CulaClient() as client:
        if args.sink_id:
            try:
                sink_id = UUID(args.sink_id)
            except ValueError:
                print(f"Invalid UUID: {args.sink_id!r}", file=sys.stderr)
                return 1
        else:
            ids = client.list_sinks()
            if not ids:
                print("No sinks returned by the API.", file=sys.stderr)
                return 1
            sink_id = ids[0]

        sink = client.get_sink(sink_id)

    G = build_entity_graph(sink)
    summary = graph_summary(G)

    print(sink_title(sink))
    print(f"  sink id: {sink.id}")
    print(f"  nodes:   {summary['nodes']}")
    print(f"  edges:   {summary['edges']}")
    print("  nodes by kind:")
    for kind, n in summary["nodes_by_kind"].items():
        print(f"    {kind}: {n}")

    if args.output:
        _write_graph(G, args.output, args.format)
        print(f"Wrote {args.format} → {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
