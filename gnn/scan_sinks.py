#!/usr/bin/env python3
"""
Run a trained :class:`gnn.model.SinkGraphVAE` on sink graph JSON files and report
structural / feature reconstruction outliers.

Uses the vocabulary stored in the checkpoint (must match training). Graphs with
**elevated** aggregate reconstruction vs the batch are highlighted; details list
the highest-scoring edges and nodes from :func:`gnn.model.anomaly_report`.

Example::

    python gnn/scan_sinks.py --data gnn/data/sinks \\
        --checkpoint gnn/checkpoints/sink_vae.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gnn.model import SinkGraphVAE, anomaly_report  # noqa: E402
from gnn.train import json_to_data  # noqa: E402


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile for p in [0, 1]."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = max(0.0, min(1.0, p))
    sv = sorted(float(v) for v in values)
    rank = (len(sv) - 1) * q
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sv[lo]
    weight = rank - lo
    return sv[lo] * (1.0 - weight) + sv[hi] * weight


def _styles() -> tuple[str, str, str, str, str]:
    if not sys.stdout.isatty():
        return ("", "", "", "", "")
    return (
        "\033[1m",  # bold
        "\033[33m",  # yellow
        "\033[31m",  # red
        "\033[2m",  # dim
        "\033[0m",  # reset
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sinks",
        help="Directory of sink graph JSON files",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "sink_vae.pt",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--top-k-edges",
        type=int,
        default=12,
        help="Edges listed per sink in detailed sections",
    )
    p.add_argument("--top-k-nodes", type=int, default=8)
    p.add_argument(
        "--alert-fraction",
        type=float,
        default=0.2,
        help="Flag the worst fraction of graphs by aggregate reconstruction loss (0–0.5).",
    )
    p.add_argument(
        "--min-graph-loss",
        type=float,
        default=None,
        help="Also flag any graph with graph_reconstruction_loss >= this value.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full per-sink reports as JSON to this path",
    )
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SinkGraphVAE, dict[str, Any]]:
    try:
        ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    meta = ckpt["meta"]
    hp = ckpt["hparams"]
    model = SinkGraphVAE(
        num_node_features=hp["num_node_features"],
        num_relations=hp["num_relations"],
        edge_attr_dim=hp["edge_attr_dim"],
        hidden_dim=hp["hidden_dim"],
        latent_dim=hp["latent_dim"],
        num_gnn_layers=hp["num_gnn_layers"],
        dropout=hp.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


def main() -> None:
    args = parse_args()
    B, Y, R, D, X = _styles()
    device = torch.device(args.device)

    paths = sorted(args.data.glob("*.json"))
    if not paths:
        print(f"No *.json files in {args.data}", file=sys.stderr)
        sys.exit(1)

    model, meta = load_model(args.checkpoint, device)
    relations: list[str] = meta["relations"]
    kind_index: dict[str, int] = meta["kind_index"]
    rel_index: dict[str, int] = meta["rel_index"]

    reports: list[dict[str, Any]] = []
    losses: list[tuple[Path, float]] = []

    for path in paths:
        data = json_to_data(path, kind_index, rel_index)
        if data.edge_index.numel() == 0:
            print(f"{Y}SKIP{X} {path.name}: no edges", flush=True)
            continue
        data = data.to(device)
        data.node_kind = data.node_kinds
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
        rep = anomaly_report(
            out,
            data,
            node_ids=data.node_ids,
            relations=relations,
            top_k_edges=args.top_k_edges,
            top_k_nodes=args.top_k_nodes,
        )
        rep["file"] = str(path.resolve())
        rep["name"] = path.name
        reports.append(rep)
        losses.append((path, rep["graph_reconstruction_loss"]))

    if not reports:
        print("No graphs scored.", file=sys.stderr)
        sys.exit(1)

    all_feature_errors: list[float] = []
    for rep in reports:
        all_feature_errors.extend(
            float(v) for v in rep.get("all_node_feature_error_raw", [])
        )
    global_max_feature_error = max(all_feature_errors) if all_feature_errors else 0.0
    feature_error_reference_p95 = _percentile(all_feature_errors, 0.95)

    if feature_error_reference_p95 > 0:
        for rep in reports:
            for nd in rep.get("anomalous_nodes", []):
                raw = float(nd.get("feature_error_raw", 0.0))
                nd["score"] = max(
                    0.0,
                    min(100.0, (raw / feature_error_reference_p95) * 100.0),
                )
    for rep in reports:
        rep.pop("all_node_feature_error_raw", None)

    losses.sort(key=lambda t: t[1], reverse=True)
    n = len(losses)
    frac = max(0.0, min(0.5, args.alert_fraction))
    k_alert = max(1, int(round(n * frac))) if frac > 0 else 0
    alert_paths: set[str] = set()
    if k_alert:
        for p, _loss in losses[:k_alert]:
            alert_paths.add(str(p.resolve()))
    if args.min_graph_loss is not None:
        for path, loss in losses:
            if loss >= args.min_graph_loss:
                alert_paths.add(str(path.resolve()))

    print(
        f"{B}Scored {n} sink graph(s){X} from {args.data} "
        f"(checkpoint {args.checkpoint.name})\n",
        flush=True,
    )

    for rep in reports:
        path = Path(rep["file"])
        loss = rep["graph_reconstruction_loss"]
        elevated = str(path.resolve()) in alert_paths

        if elevated:
            bar = f"{R}{'═' * 72}{X}"
            print(bar, flush=True)
            print(
                f"{R}{B}  ▶ ELEVATED reconstruction — check anomalies below{X}\n"
                f"  {path.name}  {D}(loss={loss:.4f}){X}",
                flush=True,
            )
            print(bar, flush=True)
            print(f"{Y}  Top anomalous edges (reconstruction){X}", flush=True)
            for e in rep["anomalous_edges"]:
                print(
                    f"    • {e['source']}  --[{e['relation']}]-->  {e['target']}"
                    f"    score={e['score']:.4f}",
                    flush=True,
                )
            print(f"{Y}  Top anomalous nodes (features){X}", flush=True)
            for nd in rep["anomalous_nodes"]:
                kind = nd.get("kind", "?")
                print(
                    f"    • {nd['id']}  (kind={kind})  score={nd['score']:.4f}",
                    flush=True,
                )
            print(flush=True)
        else:
            print(
                f"  {path.name}  {D}ok{X}  aggregate_loss={loss:.4f}",
                flush=True,
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data),
            "feature_error_reference_method": "p95_across_all_scanned_nodes",
            "feature_error_reference_p95_raw": feature_error_reference_p95,
            "global_max_feature_error_raw": global_max_feature_error,
            "alert_fraction": frac,
            "min_graph_loss": args.min_graph_loss,
            "elevated_files": sorted(alert_paths),
            "reports": reports,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{D}Wrote JSON: {args.json_out}{X}", flush=True)


if __name__ == "__main__":
    main()
