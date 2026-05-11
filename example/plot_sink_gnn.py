#!/usr/bin/env python3
"""
Run a sink graph through :class:`gnn.model.SinkGraphVAE` (checkpoint under
``gnn/checkpoints/``) and write an interactive Plotly HTML figure.

Highlights match the **top anomalous edges and nodes** from
:func:`gnn.model.anomaly_report` — the same lists ``gnn/scan_sinks.py`` prints
under **ELEVATED** sinks (``--top-k-edges`` / ``--top-k-nodes``).

**Input:** node-link JSON (e.g. from ``example/build_sink_graph.py``) or
``--fetch <sink-uuid>`` via the Cula API.

Requires ``pip install -e ".[viz]"`` and ``pip install -r gnn/requirements.txt``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any
from uuid import UUID

import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_plot_graph() -> Any:
    path = Path(__file__).resolve().parent / "plot_graph.py"
    spec = importlib.util.spec_from_file_location("_cula_plot_graph", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load plot_graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    try:
        import torch
        from gnn.model import anomaly_report
        from gnn.scan_sinks import load_model
        from gnn.train import raw_to_data
    except ImportError as e:
        print(
            "Missing GNN dependencies. Install: pip install -e \".[viz]\" && "
            "pip install -r gnn/requirements.txt\n"
            f"ImportError: {e}",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("sink.json"),
        help="Node-link JSON (default: sink.json; ignored when --fetch is set)",
    )
    parser.add_argument(
        "--fetch",
        metavar="UUID",
        help="Fetch sink from Cula API and build graph (no JSON file needed)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "gnn" / "checkpoints" / "sink_vae.pt",
        help="Trained SinkGraphVAE checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k-edges", type=int, default=12)
    parser.add_argument("--top-k-nodes", type=int, default=8)
    parser.add_argument(
        "--scan-report",
        type=Path,
        default=None,
        help=(
            "Optional scan_sinks JSON report. If set, use "
            "feature_error_reference_p95_raw (fallback: global_max_feature_error_raw) "
            "for cross-sink 0-100 node scoring."
        ),
    )
    parser.add_argument(
        "--feature-error-reference-max",
        type=float,
        default=None,
        help=(
            "Optional explicit max feature error used to map node suspicion to 0-100. "
            "Overrides --scan-report."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sink_graph_gnn.html"),
    )
    parser.add_argument(
        "--layout",
        choices=("flow", "spring", "kamada_kawai", "circular", "spectral"),
        default="flow",
    )
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--no-registry-hover", action="store_true")
    parser.add_argument("--max-registry-chars", type=int, default=12_000)
    parser.add_argument("--supply-chain-edge-hover", action="store_true")
    parser.add_argument("--max-edge-data-chars", type=int, default=4_000)
    args = parser.parse_args()

    raw: dict[str, Any]
    source_label: str

    if args.fetch:
        try:
            from cula import CulaClient
            from cula.sink_graph import build_entity_graph
        except ImportError as e:
            print(f"Cula client required for --fetch: {e}", file=sys.stderr)
            return 1
        try:
            sink_id = UUID(args.fetch)
        except ValueError:
            print(f"Invalid UUID: {args.fetch!r}", file=sys.stderr)
            return 1
        with CulaClient() as client:
            sink = client.get_sink(sink_id)
        Gb = build_entity_graph(sink)
        raw = nx.node_link_data(Gb)
        source_label = f"API sink {sink_id}"
    else:
        if not args.input.is_file():
            print(f"Input not found: {args.input}", file=sys.stderr)
            return 1
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        source_label = str(args.input.resolve())

    if not raw.get("nodes"):
        print("JSON has no nodes.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    model, meta = load_model(args.checkpoint, device)
    kind_index: dict[str, int] = meta["kind_index"]
    rel_index: dict[str, int] = meta["rel_index"]
    relations: list[str] = meta["relations"]

    data = raw_to_data(raw, kind_index, rel_index, path=source_label)
    if data.edge_index.numel() == 0:
        print("Graph has no edges; GNN scan skipped.", file=sys.stderr)
        return 1

    data = data.to(device)
    data.node_kind = data.node_kinds

    feature_error_reference_max: float | None = args.feature_error_reference_max
    if feature_error_reference_max is None and args.scan_report is not None:
        if not args.scan_report.is_file():
            print(f"Scan report not found: {args.scan_report}", file=sys.stderr)
            return 1
        scan_payload = json.loads(args.scan_report.read_text(encoding="utf-8"))
        raw_ref = scan_payload.get("feature_error_reference_p95_raw")
        if raw_ref is None:
            raw_ref = scan_payload.get("global_max_feature_error_raw")
        if raw_ref is None:
            print(
                "Scan report has no feature_error_reference_p95_raw or global_max_feature_error_raw.",
                file=sys.stderr,
            )
            return 1
        feature_error_reference_max = float(raw_ref)

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
    rep = anomaly_report(
        out,
        data,
        node_ids=data.node_ids,
        relations=relations,
        top_k_edges=args.top_k_edges,
        top_k_nodes=args.top_k_nodes,
        feature_error_reference_max=feature_error_reference_max,
    )
    loss = rep["graph_reconstruction_loss"]

    flagged_nodes = {n["id"] for n in rep["anomalous_nodes"]}
    flagged_edges = {(e["source"], e["target"]) for e in rep["anomalous_edges"]}
    gnn_node_hover = {
        n["id"]: f"<b>GNN feature error</b> score={n['score']:.4f}"
        for n in rep["anomalous_nodes"]
    }

    print(f"Source: {source_label}")
    print(f"Checkpoint: {args.checkpoint}")
    if feature_error_reference_max is not None:
        print(
            "feature_error_reference_max="
            f"{feature_error_reference_max:.6f} (cross-sink scale)"
        )
    print(f"graph_reconstruction_loss={loss:.4f}")
    print(f"Top {args.top_k_edges} anomalous edges (reconstruction):")
    for e in rep["anomalous_edges"]:
        print(
            f"  • {e['source']} --[{e['relation']}]--> {e['target']}  score={e['score']:.4f}"
        )
    print(f"Top {args.top_k_nodes} anomalous nodes (features):")
    for n in rep["anomalous_nodes"]:
        print(f"  • {n['id']}  (kind={n.get('kind', '?')})  score={n['score']:.4f}")

    pg = _load_plot_graph()
    if pg.go is None:
        print('Plotly missing. Install: pip install -e ".[viz]"', file=sys.stderr)
        return 1

    G = nx.node_link_graph(raw, directed=raw.get("directed", True))
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    pos = pg.layout_positions(
        G,
        args.layout,
        seed=args.seed,
        iterations=args.iterations,
    )
    title = (
        f"Sink graph — GNN loss={loss:.4f} · "
        f"top-{args.top_k_nodes} nodes & top-{args.top_k_edges} edges highlighted"
    )
    fig, focus_ctx = pg.build_figure(
        G,
        pos,
        max_registry_chars=args.max_registry_chars,
        show_registry_hover=not args.no_registry_hover,
        show_supply_chain_edge_hover=args.supply_chain_edge_hover,
        max_edge_data_chars=args.max_edge_data_chars,
        figure_title=title,
        flagged_nodes=flagged_nodes,
        flagged_edges=flagged_edges,
        gnn_node_hover_extra=gnn_node_hover,
    )
    fig.write_html(
        args.output,
        include_plotlyjs="cdn",
        config=dict(scrollZoom=True, displaylogo=False),
        post_script=pg._focus_post_script(focus_ctx),
    )
    print(f"Wrote {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {args.output}")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
