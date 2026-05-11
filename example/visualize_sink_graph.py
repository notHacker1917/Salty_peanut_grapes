#!/usr/bin/env python3
"""
Fetch a sink and draw a flow-style graph: the sink sits on the far right and
related entities spread left in breadth-first layers (undirected distance from sink).

Requires: pip install -e ".[viz]"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from statistics import median
from typing import Any
from uuid import UUID

from cula import CulaClient
from cula.sink_graph import build_entity_graph, sink_title as _sink_title

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:
    plt = None  # type: ignore[assignment]
    nx = None  # type: ignore[assignment]


_KIND_COLOR: dict[str, str] = {
    "sink": "#c62828",
    "org": "#1565c0",
    "site": "#2e7d32",
    "material": "#6a1b9a",
    "container": "#ef6c00",
    "event": "#00838f",
    "lca_db": "#5d4037",
    "lca_activity": "#78909c",
}


def _find_sink_node(G: "nx.DiGraph") -> Any:
    sinks = [n for n in G.nodes if G.nodes[n].get("kind") == "sink"]
    if len(sinks) != 1:
        raise ValueError("Graph must contain exactly one sink node")
    return sinks[0]


def _flow_positions_left_to_right(G: "nx.DiGraph", sink_node: Any) -> dict[Any, tuple[float, float]]:
    """
    Layer nodes by undirected graph distance from the sink: sink is the rightmost
    column; everything else spreads left in breadth-first waves (flow-style layout).
    """
    undirected = G.to_undirected()
    dist: dict[Any, int] = {}
    q: deque[Any] = deque([sink_node])
    dist[sink_node] = 0
    while q:
        n = q.popleft()
        for nb in undirected.neighbors(n):
            if nb not in dist:
                dist[nb] = dist[n] + 1
                q.append(nb)
    max_d = max(dist.values()) if dist else 0
    for n in G.nodes:
        if n not in dist:
            dist[n] = max_d + 1
    max_d = max(dist.values())

    layers: dict[int, list[Any]] = defaultdict(list)
    for n, d in dist.items():
        layers[d].append(n)

    dx = 3.0
    dy = 0.95

    def sort_key(n: Any) -> tuple[str, str]:
        return (str(G.nodes[n].get("kind", "")), str(G.nodes[n].get("label", "")))

    for d in layers:
        layers[d].sort(key=sort_key)

    # Barycentric passes: order within each layer by median index of neighbors in adjacent layers
    layer_ids = sorted(layers.keys())
    index_in_layer: dict[Any, tuple[int, int]] = {}
    for d, nodes in layers.items():
        for i, n in enumerate(nodes):
            index_in_layer[n] = (d, i)

    for _ in range(6):
        for d in layer_ids:
            nodes = layers[d]
            new_order = list(nodes)
            scores: list[tuple[float, int, Any]] = []
            for i, n in enumerate(nodes):
                neighbor_indices: list[float] = []
                for nb in undirected.neighbors(n):
                    if nb in index_in_layer:
                        nd, ni = index_in_layer[nb]
                        if abs(nd - d) == 1:
                            neighbor_indices.append(float(ni))
                b = median(neighbor_indices) if neighbor_indices else float(i)
                scores.append((b, i, n))
            scores.sort(key=lambda t: (t[0], t[1]))
            new_order = [t[2] for t in scores]
            layers[d] = new_order
            for i, n in enumerate(new_order):
                index_in_layer[n] = (d, i)

    pos: dict[Any, tuple[float, float]] = {}
    for d in layer_ids:
        nodes = layers[d]
        width = max(0.0, (len(nodes) - 1) * dy)
        for i, n in enumerate(nodes):
            x = (max_d - d) * dx
            y = width / 2 - i * dy
            pos[n] = (float(x), float(y))
    return pos


def draw_graph(G: "nx.DiGraph", out_path: str, *, title: str | None) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes")

    sink_node = _find_sink_node(G)
    pos = _flow_positions_left_to_right(G, sink_node)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    layer_span = (max(ys) - min(ys)) if ys else 0.0
    fig_w = max(22.0, max(xs) - min(xs) + 10.0)
    fig_h = max(12.0, min(56.0, layer_span + 10.0))
    plt.figure(figsize=(fig_w, fig_h))
    if title:
        plt.title(title, fontsize=14, pad=12)

    n = G.number_of_nodes()
    node_size = 820 if n < 120 else max(120, 820 - (n - 120) * 3)
    font_size = 7 if n < 120 else max(4, 7 - (n - 120) // 80)

    labels: dict[Any, str] = {n_id: str(G.nodes[n_id]["label"]) for n_id in G.nodes()}
    colors = [_KIND_COLOR.get(str(G.nodes[n_id].get("kind", "")), "#546e7a") for n_id in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=node_size,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="#263238",
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_color="white")
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=12,
        edge_color="#455a64",
        alpha=0.28,
        width=0.65,
        connectionstyle="arc3,rad=0.06",
        node_size=node_size,
    )

    margin_x = 1.2
    margin_y = 2.0
    plt.xlim(min(xs) - margin_x, max(xs) + margin_x)
    plt.ylim(min(ys) - margin_y, max(ys) + margin_y)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    if nx is None or plt is None:
        print(
            'Missing graph libraries. Install with: pip install -e ".[viz]"',
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="Fetch a sink and render a name-labeled entity graph (PNG)."
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
        default="sink_graph.png",
        help="Output image path (default: sink_graph.png)",
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
    title = _sink_title(sink)
    draw_graph(G, args.output, title=title)
    print(f"Wrote {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
