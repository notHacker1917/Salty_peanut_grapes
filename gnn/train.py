#!/usr/bin/env python3
"""
Train :class:`gnn.model.SinkGraphVAE` on locally stored sink JSON graphs.

1. Fetch data: ``python gnn/fetch_sinks.py`` (from repo root).
2. Train: ``python gnn/train.py --data gnn/data/sinks``

Defaults favor generalization: dropout, weight decay, stronger KL (with warmup),
extra negative edges, gradient clipping, LR reduction on validation plateau,
and early stopping with reload of the best validation checkpoint (when a
validation split exists).

Optional: ``--report-one <file.json>`` runs a forward pass and prints structural
anomaly candidates for that graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gnn.model import SinkGraphVAE, anomaly_report, vae_loss  # noqa: E402


def _kl_weight(epoch: int, beta_kl: float, warmup_epochs: int) -> float:
    """Linear KL ramp for the first ``warmup_epochs`` (1-based epochs)."""
    if warmup_epochs <= 0:
        return beta_kl
    t = min(epoch, warmup_epochs)
    return beta_kl * (t / warmup_epochs)


def _edge_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("edges") or payload.get("links") or []


def collect_vocab(paths: list[Path]) -> tuple[list[str], list[str]]:
    kinds: set[str] = set()
    rels: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for n in data.get("nodes", []):
            kinds.add(str(n.get("kind", "unknown")))
        for e in _edge_records(data):
            rels.add(str(e.get("relation", "unknown")))
    return sorted(kinds), sorted(rels)


def raw_to_data(
    raw: dict[str, Any],
    kind_index: dict[str, int],
    rel_index: dict[str, int],
    *,
    path: str = "",
) -> Data:
    """Build PyG :class:`Data` from a NetworkX-style node-link dict (``nodes`` / ``edges`` or ``links``)."""
    nodes = raw.get("nodes", [])
    id2i: dict[str, int] = {}
    node_ids: list[str] = []
    kinds: list[str] = []
    for i, n in enumerate(nodes):
        nid = str(n["id"])
        id2i[nid] = i
        node_ids.append(nid)
        kinds.append(str(n.get("kind", "unknown")))

    num_k = len(kind_index)
    x = torch.zeros(len(nodes), num_k, dtype=torch.float32)
    for i, k in enumerate(kinds):
        j = kind_index.get(k, 0)
        x[i, j] = 1.0

    # structural hints: in/out degree (log-scaled)
    indeg = torch.zeros(len(nodes))
    outdeg = torch.zeros(len(nodes))
    edges_src: list[int] = []
    edges_dst: list[int] = []
    rel_ids: list[int] = []
    rel_str: list[str] = []

    for e in _edge_records(raw):
        s, t = str(e["source"]), str(e["target"])
        if s not in id2i or t not in id2i:
            continue
        si, ti = id2i[s], id2i[t]
        edges_src.append(si)
        edges_dst.append(ti)
        rname = str(e.get("relation", "unknown"))
        rel_str.append(rname)
        rel_ids.append(rel_index.get(rname, 0))
        outdeg[si] += 1
        indeg[ti] += 1

    deg_feat = torch.stack(
        [torch.log1p(indeg), torch.log1p(outdeg)], dim=-1
    ).float()
    x = torch.cat([x, deg_feat], dim=-1)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    num_r = len(rel_index)
    edge_attr = torch.zeros(edge_index.size(1), num_r, dtype=torch.float32)
    for i, rname in enumerate(rel_str):
        j = rel_index.get(rname, 0)
        edge_attr[i, j] = 1.0

    edge_relation = torch.tensor(rel_ids, dtype=torch.long)

    d = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_relation=edge_relation,
    )
    d.path = path
    d.node_ids = node_ids
    d.node_kinds = kinds
    return d


def json_to_data(
    path: Path, kind_index: dict[str, int], rel_index: dict[str, int]
) -> Data:
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw_to_data(raw, kind_index, rel_index, path=str(path))
    return data


def load_dataset(
    data_dir: Path,
) -> tuple[list[Data], dict[str, Any]]:
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No *.json under {data_dir}")
    kinds, rels = collect_vocab(paths)
    kind_index = {k: i for i, k in enumerate(kinds)}
    rel_index = {r: i for i, r in enumerate(rels)}
    graphs = [json_to_data(p, kind_index, rel_index) for p in paths]
    graphs = [g for g in graphs if g.edge_index.numel() > 0]
    if not graphs:
        raise ValueError(
            f"No graphs with edges under {data_dir} (after parsing)."
        )
    meta = {
        "kinds": kinds,
        "relations": rels,
        "kind_index": kind_index,
        "rel_index": rel_index,
    }
    return graphs, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sinks",
        help="Directory of sink graph JSON files",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--latent", type=int, default=24)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--neg-ratio", type=float, default=2.0)
    p.add_argument(
        "--beta-kl",
        type=float,
        default=0.4,
        help="Target KL weight after warmup (stronger prior reduces memorization).",
    )
    p.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=25,
        help="Linearly ramp KL from 0 to --beta-kl; 0 disables warmup.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max grad norm (0 = disabled).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="Stop if validation loss does not improve (0 = disabled).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum val improvement to reset early-stopping patience.",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=8,
        help="Do not early-stop before this many epochs.",
    )
    p.add_argument(
        "--sched-patience",
        type=int,
        default=6,
        help="ReduceLROnPlateau patience (only used when a validation set exists).",
    )
    p.add_argument("--sched-factor", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "sink_vae.pt",
    )
    p.add_argument(
        "--report-one",
        type=Path,
        default=None,
        help="After training, load this JSON and print anomaly_report",
    )
    return p.parse_args()


def train() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    graphs, meta = load_dataset(args.data)
    random.shuffle(graphs)
    vf = max(0.0, min(0.5, args.val_fraction))
    if len(graphs) == 1:
        train_g, val_g = graphs, []
    else:
        n_val = max(1, int(round(len(graphs) * vf)))
        n_val = min(n_val, len(graphs) - 1)
        n_train = len(graphs) - n_val
        train_g = graphs[:n_train]
        val_g = graphs[n_train:]

    num_node_features = train_g[0].x.size(1)
    num_relations = len(meta["relations"])
    edge_attr_dim = num_relations

    loader = DataLoader(
        train_g,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_g, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = SinkGraphVAE(
        num_node_features=num_node_features,
        num_relations=num_relations,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        num_gnn_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if val_g:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.sched_factor,
            patience=args.sched_patience,
            min_lr=1e-6,
        )

    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stall = 0

    for epoch in range(1, args.epochs + 1):
        beta_eff = _kl_weight(epoch, args.beta_kl, args.beta_warmup_epochs)

        model.train()
        train_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index, batch.edge_attr)

            neg_index = batched_negative_sampling(
                batch.edge_index,
                batch.batch,
                num_neg_samples=args.neg_ratio,
            )
            neg_logits, _ = model.decode_edges(out.z, neg_index)

            loss, _ = vae_loss(
                out,
                batch.x,
                batch.edge_relation,
                neg_logits,
                beta_kl=beta_eff,
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(loader))

        model.eval()
        val_loss = 0.0
        if val_g:
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr)
                    neg_index = batched_negative_sampling(
                        batch.edge_index,
                        batch.batch,
                        num_neg_samples=args.neg_ratio,
                    )
                    neg_logits, _ = model.decode_edges(out.z, neg_index)
                    loss, _ = vae_loss(
                        out,
                        batch.x,
                        batch.edge_relation,
                        neg_logits,
                        beta_kl=beta_eff,
                    )
                    val_loss += float(loss.item())
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            improved = val_loss < best_val - args.early_stop_min_delta
            if improved:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1

            if (
                val_g
                and args.early_stop_patience > 0
                and epoch >= args.min_epochs
                and stall >= args.early_stop_patience
            ):
                print(
                    f"early stopping at epoch {epoch} (best val_loss={best_val:.4f})",
                    flush=True,
                )
                break
        else:
            val_loss = float("nan")

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}  beta_kl={beta_eff:.4f}  lr={lr_now:.2e}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": meta,
            "hparams": {
                "hidden_dim": args.hidden,
                "latent_dim": args.latent,
                "num_gnn_layers": args.layers,
                "dropout": args.dropout,
                "num_node_features": num_node_features,
                "num_relations": num_relations,
                "edge_attr_dim": edge_attr_dim,
                "weight_decay": args.weight_decay,
                "beta_kl": args.beta_kl,
                "beta_warmup_epochs": args.beta_warmup_epochs,
            },
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint: {args.checkpoint}")

    if args.report_one is not None:
        report_path = args.report_one
        data = json_to_data(
            report_path, meta["kind_index"], meta["rel_index"]
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            data.node_kind = data.node_kinds
            rep = anomaly_report(
                out,
                data,
                node_ids=data.node_ids,
                relations=meta["relations"],
            )
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    train()
