"""Candidate-ranking BC for TG-FPS (listwise CE over exact legal candidates)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train_rl import build_policy, load_policy
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def candidate_ranking_loss(
    logits: torch.Tensor,
    best_index: int,
    score_vectors: torch.Tensor | None = None,
    *,
    pairwise_weight: float = 0.3,
    delta_weight: float = 0.1,
    feas_aux_weight: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Listwise CE + pairwise margin + optional lex-delta regression + feas aux."""
    target = torch.tensor(best_index, device=logits.device, dtype=torch.long)
    listwise = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss = listwise
    metrics = {"listwise_ce": float(listwise.detach())}
    if score_vectors is not None and score_vectors.numel() > 0 and logits.numel() > 1:
        best = logits[best_index]
        margin = 0.1
        pairwise = torch.tensor(0.0, device=logits.device)
        count = 0
        for i in range(logits.numel()):
            if i == best_index:
                continue
            pairwise = pairwise + F.relu(margin - (best - logits[i]))
            count += 1
        if count:
            pairwise = pairwise / count
            loss = loss + pairwise_weight * pairwise
            metrics["pairwise_margin"] = float(pairwise.detach())
        if score_vectors.dim() == 2 and score_vectors.size(0) == logits.numel():
            obj = score_vectors[:, -1]
            obj_n = (obj - obj.mean()) / (obj.std() + 1e-6)
            pred = -logits
            pred_n = (pred - pred.mean()) / (pred.std() + 1e-6)
            delta = F.mse_loss(pred_n, obj_n)
            loss = loss + delta_weight * delta
            metrics["delta_regression"] = float(delta.detach())
            hard = score_vectors[:, 0]
            soft_target = torch.softmax(-hard, dim=0)
            log_p = F.log_softmax(logits, dim=0)
            feas = -(soft_target * log_p).sum()
            loss = loss + feas_aux_weight * feas
            metrics["feas_aux"] = float(feas.detach())
    metrics["loss"] = float(loss.detach())
    return loss, metrics


class _CandidateAuxHead(nn.Module):
    """Fast ranking head over [acting features | score vector | node id emb]."""

    def __init__(self, feat_dim: int = 32, score_dim: int = 7, emb_dim: int = 16):
        super().__init__()
        self.node_emb = nn.Embedding(512, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + score_dim + emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        acting: torch.Tensor,
        scores: torch.Tensor,
        nodes: torch.Tensor,
    ) -> torch.Tensor:
        # acting: [F], scores: [K,S], nodes: [K]
        k = scores.size(0)
        a = acting.unsqueeze(0).expand(k, -1)
        if a.size(-1) < 32:
            a = F.pad(a, (0, 32 - a.size(-1)))
        else:
            a = a[..., :32]
        if scores.size(-1) < 7:
            scores = F.pad(scores, (0, 7 - scores.size(-1)))
        else:
            scores = scores[..., :7]
        e = self.node_emb(nodes.clamp(0, 511))
        x = torch.cat([a, scores, e], dim=-1)
        return self.mlp(x).squeeze(-1)


def train_candidate_ranker(
    config: dict,
    decisions: list[dict],
    *,
    init_checkpoint: str | Path | None = None,
    epochs: int = 5,
    device: str | torch.device = "cpu",
) -> dict:
    """Train ranking head (+ optional pointer encode) via listwise CE."""
    device = torch.device(device)
    portions = list((config.get("problem") or {}).get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    if init_checkpoint and Path(init_checkpoint).is_file():
        policy = load_policy(init_checkpoint, device=device)
    else:
        policy = build_policy(config, portions).to(device)

    if not hasattr(policy, "candidate_aux_head"):
        policy.add_module("candidate_aux_head", _CandidateAuxHead().to(device))

    opt = torch.optim.Adam(
        policy.parameters(),
        lr=float((config.get("behavior_cloning") or {}).get("learning_rate", 1e-4)),
    )
    rng = random.Random(int(config.get("seed", 42)))
    history: list[dict[str, Any]] = []
    if not decisions:
        raise RuntimeError("No candidate decisions for ranking BC")

    max_per_epoch = int((config.get("pipeline") or {}).get("ranker_max_per_epoch", 64))
    policy.train()
    for epoch in range(epochs):
        rng.shuffle(decisions)
        losses = []
        n_skipped = 0
        for dec in decisions[: min(len(decisions), max_per_epoch)]:
            nodes = list(dec.get("candidate_nodes") or [])
            if len(nodes) < 2:
                n_skipped += 1
                continue
            best = int(dec["best_candidate_index"])
            best = max(0, min(best, len(nodes) - 1))
            scores_t = torch.tensor(
                dec.get("score_vectors") or [[0.0] * 7] * len(nodes),
                dtype=torch.float32,
                device=device,
            )
            if scores_t.size(0) != len(nodes):
                scores_t = torch.zeros(len(nodes), 7, device=device)
            aef = torch.tensor(
                (dec.get("acting_ev_features") or [0.0] * 32)[:32],
                dtype=torch.float32,
                device=device,
            )
            if aef.numel() < 32:
                aef = F.pad(aef, (0, 32 - aef.numel()))
            node_t = torch.tensor(nodes, dtype=torch.long, device=device)
            logits = policy.candidate_aux_head(aef, scores_t, node_t)
            loss, _ = candidate_ranking_loss(logits, best, scores_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses) if losses else 0.0)
        history.append({"epoch": epoch + 1, "loss": mean_loss, "n_skipped": n_skipped})
        LOGGER.info(
            "Candidate-ranker epoch %d/%d loss=%.4f skipped=%d",
            epoch + 1, epochs, mean_loss, n_skipped,
        )

    out_dir = Path((config.get("behavior_cloning") or {}).get("output_dir", "outputs/bc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "candidate_ranker_best.pt"
    last_path = out_dir / "candidate_ranker_last.pt"
    raw_policy = getattr(policy, "_orig_mod", policy)
    full_state = raw_policy.state_dict() if hasattr(raw_policy, "state_dict") else policy.state_dict()
    aux_state = {
        k.replace("candidate_aux_head.", "", 1): v
        for k, v in full_state.items()
        if k.startswith("candidate_aux_head.")
    }
    model_state = {k: v for k, v in full_state.items() if not k.startswith("candidate_aux_head.")}
    payload = {
        "architecture_version": "exact_smdp_dynamic_v1",
        "model_state": model_state,
        "candidate_aux_head": aux_state or policy.candidate_aux_head.state_dict(),
        "model_config": config["model"],
        "charging_portions": portions,
        "candidate_ranker": True,
        "history": history,
    }
    torch.save(payload, best_path)
    torch.save(payload, last_path)
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
        "n_decisions": len(decisions),
    }
