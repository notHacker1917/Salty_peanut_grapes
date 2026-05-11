#!/usr/bin/env python3
"""
Embedding-based sink anomaly detection.

This scanner loads a trained ``gnn`` checkpoint, encodes each sink graph into a
single graph embedding (mean of node-level encoder means ``mu``), and computes
an outlier score in embedding space using regularized Mahalanobis distance.

Higher score => farther from the cohort center => more anomalous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gnn.model import SinkGraphVAE  # noqa: E402
from gnn.train import json_to_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "gnn" / "data" / "sinks",
        help="Directory containing sink graph JSON files.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "gnn"
        / "checkpoints"
        / "sink_vae.pt",
        help="Trained SinkGraphVAE checkpoint.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--cov-reg",
        type=float,
        default=1e-3,
        help="Covariance ridge regularization factor.",
    )
    p.add_argument(
        "--alert-fraction",
        type=float,
        default=0.2,
        help="Highlight top fraction by outlier score (0-0.5).",
    )
    p.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Also highlight sinks with robust z-score >= threshold.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full scores as JSON.",
    )
    return p.parse_args()


def _styles() -> tuple[str, str, str, str, str]:
    if not sys.stdout.isatty():
        return ("", "", "", "", "")
    return ("\033[1m", "\033[33m", "\033[31m", "\033[2m", "\033[0m")


def _load_checkpoint(path: Path, device: torch.device) -> tuple[SinkGraphVAE, dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

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
    return model, ckpt["meta"]


def _embed_graph(model: SinkGraphVAE, data: Any) -> torch.Tensor:
    mu, _logvar = model.encode(data.x, data.edge_index, data.edge_attr)
    return mu.mean(dim=0)


def _mahalanobis_scores(
    emb: torch.Tensor, cov_reg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (scores, center) where scores are Mahalanobis squared distances from
    a robust center (dimension-wise median).
    """
    if emb.dim() != 2:
        raise ValueError("Expected embeddings shape [n_graphs, dim].")
    n, d = emb.shape
    if n < 2:
        # Not enough samples for covariance; fallback to zeros.
        return torch.zeros(n, device=emb.device), emb[0]

    center = emb.median(dim=0).values
    diff = emb - center
    cov = diff.T @ diff / max(n - 1, 1)
    # Scale regularizer by average variance to stay unit-aware.
    avg_var = torch.trace(cov) / max(d, 1)
    reg = max(cov_reg, 1e-8) * (avg_var + 1e-8)
    cov_reg_mat = cov + torch.eye(d, device=emb.device) * reg
    inv_cov = torch.linalg.pinv(cov_reg_mat)
    scores = (diff @ inv_cov * diff).sum(dim=1)
    return scores, center


def _robust_z(scores: torch.Tensor) -> torch.Tensor:
    med = scores.median()
    mad = (scores - med).abs().median()
    # 0.6745 scales MAD to std for Gaussian data.
    scale = (mad / 0.6745).clamp_min(1e-8)
    return (scores - med) / scale


def main() -> None:
    args = parse_args()
    B, Y, R, D, X = _styles()

    device = torch.device(args.device)
    model, meta = _load_checkpoint(args.checkpoint, device)
    kind_index = meta["kind_index"]
    rel_index = meta["rel_index"]

    paths = sorted(args.data.glob("*.json"))
    if not paths:
        print(f"No *.json files in {args.data}", file=sys.stderr)
        raise SystemExit(1)

    rows: list[dict[str, Any]] = []
    emb_list: list[torch.Tensor] = []
    for p in paths:
        d = json_to_data(p, kind_index, rel_index)
        if d.edge_index.numel() == 0:
            continue
        d = d.to(device)
        with torch.no_grad():
            e = _embed_graph(model, d)
        emb_list.append(e)
        rows.append(
            {
                "file": str(p.resolve()),
                "name": p.name,
                "nodes": int(d.x.size(0)),
                "edges": int(d.edge_index.size(1)),
            }
        )

    if not rows:
        print("No usable sink graphs found (all empty?).", file=sys.stderr)
        raise SystemExit(1)

    emb = torch.stack(emb_list, dim=0)
    scores, _center = _mahalanobis_scores(emb, cov_reg=args.cov_reg)
    z = _robust_z(scores)

    for i, row in enumerate(rows):
        row["embedding_outlier_score"] = float(scores[i].item())
        row["robust_z"] = float(z[i].item())

    ranked = sorted(rows, key=lambda r: r["embedding_outlier_score"], reverse=True)

    frac = max(0.0, min(0.5, args.alert_fraction))
    top_n = max(1, int(round(len(ranked) * frac))) if frac > 0 else 0
    top_alert = {r["file"] for r in ranked[:top_n]}
    z_alert = {r["file"] for r in ranked if r["robust_z"] >= args.z_threshold}
    elevated = top_alert | z_alert

    print(
        f"{B}Embedding outlier scan: {len(ranked)} sinks{X} "
        f"{D}(checkpoint={args.checkpoint.name}){X}",
        flush=True,
    )
    print(
        f"{D}Rule: top {frac:.0%} by score OR robust_z >= {args.z_threshold:.2f}{X}\n",
        flush=True,
    )

    for r in ranked:
        file_path = r["file"]
        score = r["embedding_outlier_score"]
        rz = r["robust_z"]
        label = (
            f"{R}{B}ANOMALY{X}" if file_path in elevated else f"{Y}normal-ish{X}"
        )
        print(
            f"{label}  {r['name']}  "
            f"score={score:.4f}  z={rz:.2f}  "
            f"{D}(nodes={r['nodes']}, edges={r['edges']}){X}",
            flush=True,
        )

    if not elevated:
        print(f"\n{Y}No strong embedding outliers found with current thresholds.{X}")

    if args.json_out is not None:
        payload = {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data),
            "cov_reg": args.cov_reg,
            "alert_fraction": frac,
            "z_threshold": args.z_threshold,
            "elevated_files": sorted(elevated),
            "scores_desc": ranked,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{D}Wrote JSON report: {args.json_out}{X}", flush=True)


if __name__ == "__main__":
    main()
