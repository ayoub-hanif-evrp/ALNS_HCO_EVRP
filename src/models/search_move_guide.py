"""Transformer search-move guide: scores ALNS operators and destroy targets.

This is intentionally NOT node-level BC / construction policy.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = (
    "random_customers",
    "critical_tw",
    "route_removal",
    "relocate",
    "swap",
    "two_opt",
    "station_clean",
)
OP_TO_IDX = {o: i for i, o in enumerate(OPS)}
N_OPS = len(OPS)
NODE_FEAT_DIM = 12
GLOBAL_FEAT_DIM = 16


def build_node_features(instance, routes=None) -> torch.Tensor:
    """Static+served node features [N, NODE_FEAT_DIM]."""
    served = set()
    if routes is not None:
        for nodes, _ in routes:
            for n in nodes:
                if instance.nodes[n].is_customer:
                    served.add(int(n))
    rows = []
    for i, n in enumerate(instance.nodes):
        due = float(getattr(n, "due_time", 0.0) or 0.0)
        ready = float(getattr(n, "ready_time", 0.0) or 0.0)
        service = float(getattr(n, "service_time", 0.0) or 0.0)
        demand = float(getattr(n, "demand", 0.0) or 0.0)
        x = float(getattr(n, "x", 0.0) or 0.0)
        y = float(getattr(n, "y", 0.0) or 0.0)
        rows.append([
            x / 100.0,
            y / 100.0,
            demand / 50.0,
            ready / 1000.0,
            due / 1000.0,
            service / 100.0,
            1.0 if n.is_depot else 0.0,
            1.0 if n.is_customer else 0.0,
            1.0 if n.is_charging_station else 0.0,
            1.0 if i in served else 0.0,
            (due - ready) / 1000.0,
            float(len(served)) / max(float(instance.num_customers), 1.0),
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_global_features(instance, routes, score=None) -> torch.Tensor:
    """Compact solution-level features [GLOBAL_FEAT_DIM]."""
    n_cust = max(int(instance.num_customers), 1)
    served = set()
    route_lens = []
    dues = []
    for nodes, _ in routes:
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        route_lens.append(len(custs))
        for c in custs:
            served.add(c)
            dues.append(float(instance.nodes[c].due_time))
    used = sum(1 for L in route_lens if L > 0)
    mean_len = (sum(route_lens) / max(used, 1)) if route_lens else 0.0
    mean_due = (sum(dues) / len(dues)) if dues else 0.0
    hard = float(getattr(score, "hard_violation_count", 0) or 0)
    unserved = float(n_cust - len(served))
    dist = float(getattr(score, "total_distance", 0.0) or 0.0)
    chg = float(getattr(score, "total_charging_time", 0.0) or 0.0)
    dur = float(getattr(score, "total_duration", 0.0) or 0.0)
    obj = float(getattr(score, "paper_objective", 0.0) or 0.0)
    return torch.tensor([
        len(served) / n_cust,
        unserved / n_cust,
        used / max(float(len(routes)), 1.0),
        mean_len / 20.0,
        mean_due / 1000.0,
        hard / 10.0,
        dist / 5000.0,
        chg / 500.0,
        dur / 10000.0,
        obj / 1e5,
        float(_is_type1(instance)),
        float(_class_id(instance) == "C"),
        float(_class_id(instance) == "R"),
        float(_class_id(instance) == "RC"),
        float(n_cust) / 100.0,
        float(len(routes)) / 50.0,
    ], dtype=torch.float32)


def _is_type1(instance) -> bool:
    name = str(getattr(instance, "instance_name", "") or "")
    # Schneider: *101-*109 type1-ish early TW; *201-*210 type2. Use digit after class letter.
    core = name.split("_")[0]
    digits = "".join(ch for ch in core if ch.isdigit())
    if not digits:
        return False
    return int(digits) < 200


def _class_id(instance) -> str:
    name = str(getattr(instance, "instance_name", "") or "").lower()
    if name.startswith("rc"):
        return "RC"
    if name.startswith("c"):
        return "C"
    if name.startswith("r"):
        return "R"
    return "?"


class SearchMoveGuideNet(nn.Module):
    """Small Transformer over nodes + heads for ops / customers / routes."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_proj = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_FEAT_DIM, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.op_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, N_OPS),
        )
        self.customer_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.route_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def encode(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [N, F] or [B, N, F]
        squeeze = node_features.dim() == 2
        if squeeze:
            node_features = node_features.unsqueeze(0)
        h = self.node_proj(node_features)
        h = self.encoder(h)
        return h.squeeze(0) if squeeze else h

    def forward_ops(self, node_emb: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        if node_emb.dim() == 2:
            pooled = node_emb.mean(dim=0)
        else:
            pooled = node_emb.mean(dim=1)
        g = self.global_proj(global_feat)
        if pooled.dim() == 1:
            x = torch.cat([pooled, g], dim=-1)
        else:
            x = torch.cat([pooled, g], dim=-1)
        return self.op_head(x)

    def forward_customers(
        self,
        node_emb: torch.Tensor,
        global_feat: torch.Tensor,
        customer_ids: Sequence[int],
    ) -> torch.Tensor:
        g = self.global_proj(global_feat)
        if not customer_ids:
            return torch.zeros(0, device=node_emb.device)
        idxs = torch.tensor(list(customer_ids), dtype=torch.long, device=node_emb.device)
        h = node_emb.index_select(0, idxs)
        g_exp = g.unsqueeze(0).expand(h.size(0), -1)
        return self.customer_head(torch.cat([h, g_exp], dim=-1)).squeeze(-1)

    def forward_routes(
        self,
        node_emb: torch.Tensor,
        global_feat: torch.Tensor,
        route_feats: torch.Tensor,
    ) -> torch.Tensor:
        """route_feats: [R, embedding_dim + 4]."""
        g = self.global_proj(global_feat)
        pooled = route_feats[:, : self.embedding_dim]
        extra = route_feats[:, self.embedding_dim :]
        g_exp = g.unsqueeze(0).expand(route_feats.size(0), -1)
        return self.route_head(torch.cat([pooled, g_exp, extra], dim=-1)).squeeze(-1)


def listwise_ce(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    target = torch.tensor([int(target_idx)], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.unsqueeze(0), target)


def pairwise_rank_loss(logits: torch.Tensor, positive: Sequence[int], negative: Sequence[int]) -> torch.Tensor:
    if not positive or not negative or logits.numel() == 0:
        return logits.sum() * 0.0
    pos = logits[list(positive)].mean()
    neg = logits[list(negative)].mean()
    return F.softplus(neg - pos)
