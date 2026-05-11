"""
Graph variational autoencoder for sink graphs.

The encoder is edge-aware (GINEConv). Latents are **per-node** Gaussians; the
decoder reconstructs node features and, for each edge, presence (real edge vs
negative sample) plus relation type. High reconstruction loss on edges or nodes
flags structural anomalies relative to the training distribution.

Typical use after training:

1. Run ``forward`` on a :class:`torch_geometric.data.Data` instance.
2. Call :func:`anomaly_report` to obtain the highest-scoring edges and nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models.mlp import MLP


def _map_feature_error_to_suspicion(
    feature_error: Tensor,
    *,
    reference_max_error: float | None = None,
) -> Tensor:
    """
    Map raw feature reconstruction error to 0..100 suspiciousness.

    The mapping is anchored at 0 error and scaled by a max reference:
    - 0 means not suspicious (no feature reconstruction error)
    - 100 means at/above the chosen maximum reference error

    If ``reference_max_error`` is not provided, the current graph max is used.
    """
    if feature_error.numel() == 0:
        return feature_error
    if reference_max_error is None:
        max_ref = torch.max(feature_error)
    else:
        max_ref = torch.tensor(
            float(reference_max_error),
            dtype=feature_error.dtype,
            device=feature_error.device,
        )
    if torch.isclose(max_ref, torch.zeros_like(max_ref)):
        return torch.zeros_like(feature_error)
    suspiciousness = (feature_error / max_ref) * 100.0
    return suspiciousness.clamp(0.0, 100.0)


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Average KL(q(z|x) || N(0,I)) over nodes."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


@dataclass
class SinkGraphVAEResult:
    """Outputs of :meth:`SinkGraphVAE.forward`."""

    z: Tensor
    mu: Tensor
    logvar: Tensor
    x_recon: Tensor
    edge_presence_logit: Tensor
    edge_relation_logits: Tensor


class SinkGraphVAE(nn.Module):
    """
    Variational GNN: GINE encoder, Gaussian latents per node, MLP decoders for
    features and edges (presence + relation class on positive/observed edges).
    """

    def __init__(
        self,
        num_node_features: int,
        num_relations: int,
        edge_attr_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_gnn_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.latent_dim = latent_dim
        self.dropout = float(dropout)

        dims = [hidden_dim] * num_gnn_layers
        convs: list[GINEConv] = []
        norms: list[nn.BatchNorm1d] = []
        in_dim = num_node_features
        for out_dim in dims:
            nn_g = MLP([in_dim, out_dim, out_dim], norm=None, act="relu")
            convs.append(GINEConv(nn_g, train_eps=True, edge_dim=edge_attr_dim))
            norms.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.to_mu = nn.Linear(in_dim, latent_dim)
        self.to_logvar = nn.Linear(in_dim, latent_dim)
        self.node_decoder = nn.Linear(latent_dim, num_node_features)

        edge_in = 2 * latent_dim
        self.edge_trunk = MLP(
            [edge_in, hidden_dim, hidden_dim],
            norm=None,
            act="relu",
        )
        self.edge_presence = nn.Linear(hidden_dim, 1)
        self.edge_relation = nn.Linear(hidden_dim, num_relations)

    def encode(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        h = x
        for conv, bn in zip(self.convs, self.norms, strict=True):
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.to_mu(h), self.to_logvar(h)

    def decode_edges(self, z: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        pair = torch.cat([z[row], z[col]], dim=-1)
        if self.dropout > 0:
            pair = F.dropout(pair, p=self.dropout, training=self.training)
        trunk = self.edge_trunk(pair)
        return self.edge_presence(trunk), self.edge_relation(trunk)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> SinkGraphVAEResult:
        mu, logvar = self.encode(x, edge_index, edge_attr)
        z = reparameterize(mu, logvar)
        x_recon = self.node_decoder(z)
        presence_logit, relation_logits = self.decode_edges(z, edge_index)
        return SinkGraphVAEResult(
            z=z,
            mu=mu,
            logvar=logvar,
            x_recon=x_recon,
            edge_presence_logit=presence_logit.squeeze(-1),
            edge_relation_logits=relation_logits,
        )


def vae_loss(
    out: SinkGraphVAEResult,
    x: Tensor,
    edge_relation_index: Tensor,
    neg_presence_logit: Tensor,
    beta_kl: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_rel: float = 1.0,
    lambda_feat: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Combine ELBO-style terms.

    :param out: Forward pass on positive ``edge_index``.
    :param edge_relation_index: Integer relation id per positive edge.
    :param neg_presence_logit: Presence logits for negative edge pairs (no relation loss).
    """
    kl = kl_divergence(out.mu, out.logvar)
    feat = F.mse_loss(out.x_recon, x)

    pos_bce = F.binary_cross_entropy_with_logits(
        out.edge_presence_logit,
        torch.ones_like(out.edge_presence_logit),
    )
    neg_bce = F.binary_cross_entropy_with_logits(
        neg_presence_logit,
        torch.zeros_like(neg_presence_logit),
    )
    rel = F.cross_entropy(out.edge_relation_logits, edge_relation_index)

    total = (
        beta_kl * kl
        + lambda_feat * feat
        + lambda_edge * (pos_bce + neg_bce)
        + lambda_rel * rel
    )
    parts = {
        "loss": total,
        "kl": kl.detach(),
        "feat": feat.detach(),
        "pos_bce": pos_bce.detach(),
        "neg_bce": neg_bce.detach(),
        "rel": rel.detach(),
    }
    return total, parts


@torch.no_grad()
def anomaly_report(
    out: SinkGraphVAEResult,
    data: Any,
    node_ids: list[str],
    relations: list[str],
    top_k_edges: int = 25,
    top_k_nodes: int = 15,
    feature_error_reference_max: float | None = None,
) -> dict[str, Any]:
    """
    Rank edges and nodes by local reconstruction error (structure + features).

    ``data`` must expose ``edge_index`` and ``edge_relation`` (int per edge).
    ``node_ids[i]`` is the string id for PyG node index ``i``.
    """
    edge_index = data.edge_index
    rel_idx = data.edge_relation

    x = data.x
    feat_err = (out.x_recon - x).pow(2).sum(dim=-1)
    feat_suspicion = _map_feature_error_to_suspicion(
        feat_err,
        reference_max_error=feature_error_reference_max,
    )

    pos_bce = F.binary_cross_entropy_with_logits(
        out.edge_presence_logit,
        torch.ones_like(out.edge_presence_logit),
        reduction="none",
    )
    rel_ce = F.cross_entropy(
        out.edge_relation_logits, rel_idx, reduction="none"
    )
    edge_score = pos_bce + rel_ce

    rows, cols = edge_index[0].tolist(), edge_index[1].tolist()
    edges_ranked = sorted(
        range(edge_index.size(1)),
        key=lambda e: float(edge_score[e].item()),
        reverse=True,
    )[:top_k_edges]

    anom_edges: list[dict[str, Any]] = []
    for e in edges_ranked:
        i, j = rows[e], cols[e]
        rid = int(rel_idx[e].item())
        anom_edges.append(
            {
                "source": node_ids[i],
                "target": node_ids[j],
                "relation": relations[rid] if rid < len(relations) else str(rid),
                "score": float(edge_score[e].item()),
            }
        )

    nodes_ranked = sorted(
        range(x.size(0)),
        key=lambda n: float(feat_err[n].item()),
        reverse=True,
    )[:top_k_nodes]

    kinds = getattr(data, "node_kind", None)
    anom_nodes: list[dict[str, Any]] = []
    for n in nodes_ranked:
        entry: dict[str, Any] = {
            "id": node_ids[n],
            "score": float(feat_suspicion[n].item()),
            "feature_error_raw": float(feat_err[n].item()),
        }
        if kinds is not None:
            entry["kind"] = str(kinds[n])
        anom_nodes.append(entry)

    return {
        "anomalous_edges": anom_edges,
        "anomalous_nodes": anom_nodes,
        "all_node_feature_error_raw": feat_err.detach().cpu().tolist(),
        "max_feature_error_raw": float(torch.max(feat_err).item()),
        "graph_reconstruction_loss": float(
            feat_err.mean().item()
            + edge_score.mean().item()
        ),
    }
