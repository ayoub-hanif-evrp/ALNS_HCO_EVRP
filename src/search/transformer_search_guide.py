"""TransformerSearchMoveGuide: ALNS-facing wrapper around SearchMoveGuideNet."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch

from src.models.search_move_guide import (
    OPS,
    OP_TO_IDX,
    SearchMoveGuideNet,
    build_global_features,
    build_node_features,
)
from src.search.neural_guide import InferenceTelemetry, NeuralCheckpointError, file_sha256
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


class TransformerSearchMoveGuide:
    """Learned search-move guide (operators + removals + routes). Not node BC."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        cfg = config or {}
        ng = ((cfg.get("hybrid_search") or {}).get("neural_guidance") or {})
        path = Path(checkpoint) if checkpoint else Path(str(ng.get("checkpoint") or ""))
        if not path.is_file():
            raise NeuralCheckpointError(f"Transformer search-move checkpoint missing: {path}")
        from src.utils.seed import resolve_device

        self.config = cfg
        self.device = torch.device(device or resolve_device(cfg.get("device", "cpu")))
        self.checkpoint_path = path
        self.checkpoint_sha256 = file_sha256(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        arch = dict(payload.get("arch") or {})
        self.net = SearchMoveGuideNet(
            embedding_dim=int(arch.get("embedding_dim", 64)),
            num_heads=int(arch.get("num_heads", 4)),
            num_layers=int(arch.get("num_layers", 2)),
            ff_dim=int(arch.get("ff_dim", 128)),
            dropout=float(arch.get("dropout", 0.1)),
        ).to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.telemetry = InferenceTelemetry(
            checkpoint_path=str(path.resolve()),
            checkpoint_sha256=self.checkpoint_sha256,
            topk=5,
        )
        self._cache: dict[str, torch.Tensor] = {}
        self._context_routes = None
        self._context_score = None
        LOGGER.info(
            "TransformerSearchMoveGuide loaded %s sha256=%s...",
            path, self.checkpoint_sha256[:16],
        )

    def reset_telemetry(self) -> None:
        self.telemetry = InferenceTelemetry(
            checkpoint_path=str(self.checkpoint_path.resolve()),
            checkpoint_sha256=self.checkpoint_sha256,
            topk=5,
        )
        self._cache.clear()

    def set_solution_context(self, routes, score=None) -> None:
        self._context_routes = routes
        self._context_score = score

    @torch.no_grad()
    def _encode(self, instance, routes=None) -> tuple[torch.Tensor, torch.Tensor]:
        routes = routes if routes is not None else self._context_routes
        key = f"{getattr(instance, 'instance_name', id(instance))}:{id(routes)}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        nodes = build_node_features(instance, routes).to(self.device)
        emb = self.net.encode(nodes)
        g = build_global_features(instance, routes or [], self._context_score).to(self.device)
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.encode_calls += 1
        self._cache[key] = (emb, g)
        return emb, g

    @torch.no_grad()
    def rank_destroy_operators(
        self,
        op_names: list[str],
        context_features: Sequence[float] | None = None,
        *,
        instance=None,
        routes=None,
    ) -> list[str]:
        if instance is None:
            # Fallback preference if called without context (should be rare).
            preferred = list(OPS)
            ordered = [o for o in preferred if o in op_names] + [o for o in op_names if o not in preferred]
            self.telemetry.destroy_choices += 1
            self.telemetry.neural_decisions += 1
            return ordered
        emb, g = self._encode(instance, routes)
        t0 = time.perf_counter()
        logits = self.net.forward_ops(emb, g)
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.score_candidate_calls += 1
        scores = {op: float(logits[OP_TO_IDX[op]].item()) for op in OPS if op in op_names}
        for op in op_names:
            scores.setdefault(op, -1e9)
        ordered = sorted(op_names, key=lambda o: scores[o], reverse=True)
        self.telemetry.destroy_choices += 1
        self.telemetry.neural_decisions += 1
        return ordered

    @torch.no_grad()
    def rank_customers_for_removal(
        self,
        instance,
        customers: list[int],
        *,
        limit: int | None = None,
        routes=None,
    ) -> list[int]:
        if not customers:
            return []
        emb, g = self._encode(instance, routes)
        t0 = time.perf_counter()
        logits = self.net.forward_customers(emb, g, customers)
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.score_candidate_calls += 1
        order = torch.argsort(logits, descending=True).tolist()
        ranked = [int(customers[i]) for i in order]
        if limit is not None:
            ranked = ranked[: max(0, limit)]
        self.telemetry.removal_rankings += 1
        self.telemetry.neural_decisions += 1
        self.telemetry.candidates_generated += len(customers)
        if len(customers) > len(ranked):
            self.telemetry.candidates_avoided_by_neural_order += len(customers) - len(ranked)
        return ranked

    @torch.no_grad()
    def rank_route_indices(
        self,
        instance,
        routes: list,
        *,
        prefer_tight: bool = True,
    ) -> list[int]:
        emb, g = self._encode(instance, routes)
        feats = []
        valid = []
        for ri, (nodes, _) in enumerate(routes):
            custs = [n for n in nodes if instance.nodes[n].is_customer]
            if not custs:
                continue
            idxs = torch.tensor(custs, dtype=torch.long, device=emb.device)
            pooled = emb.index_select(0, idxs).mean(dim=0)
            dues = [float(instance.nodes[c].due_time) for c in custs]
            extra = torch.tensor(
                [
                    float(len(custs)) / 20.0,
                    (sum(dues) / len(dues)) / 1000.0,
                    1.0 if prefer_tight else 0.0,
                    -float(len(custs)) / 20.0,
                ],
                device=emb.device,
            )
            feats.append(torch.cat([pooled, extra], dim=-1))
            valid.append(ri)
        if not feats:
            return list(range(len(routes)))
        route_feats = torch.stack(feats, dim=0)
        t0 = time.perf_counter()
        logits = self.net.forward_routes(emb, g, route_feats)
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.score_candidate_calls += 1
        order = torch.argsort(logits, descending=True).tolist()
        ranked = [valid[i] for i in order]
        # Append empty routes last.
        ranked += [i for i in range(len(routes)) if i not in ranked]
        self.telemetry.route_choices += 1
        self.telemetry.neural_decisions += 1
        return ranked

    def rank_insertion_candidates(self, candidates: list[Any], *, key_fn=None) -> list[Any]:
        if not candidates:
            return []
        if key_fn is not None:
            ranked = sorted(candidates, key=key_fn)
        else:
            def _k(c):
                cust = int(getattr(c, "customer", 0))
                st = int(getattr(c, "station_before", None) is not None) + int(
                    getattr(c, "station_after", None) is not None
                )
                return (st, cust)

            ranked = sorted(candidates, key=_k)
        self.telemetry.insertion_rankings += 1
        self.telemetry.neural_decisions += 1
        self.telemetry.candidates_generated += len(candidates)
        return ranked

    def choose_elite(self, elites: list[Any], scores: list[Any] | None = None) -> int:
        if not elites:
            return 0
        if scores:
            best_i, best = 0, scores[0]
            for i, s in enumerate(scores):
                if hasattr(s, "dominates") and s.dominates(best):
                    best_i, best = i, s
            self.telemetry.elite_choices += 1
            self.telemetry.neural_decisions += 1
            return best_i
        self.telemetry.elite_choices += 1
        self.telemetry.neural_decisions += 1
        return 0


def load_search_move_guide(config: dict, *, device=None) -> TransformerSearchMoveGuide:
    ng = ((config.get("hybrid_search") or {}).get("neural_guidance") or {})
    ckpt = ng.get("checkpoint")
    if not ckpt:
        raise NeuralCheckpointError("hybrid_search.neural_guidance.checkpoint is required for P11")
    if bool(ng.get("deprecated_bc_guide", False)):
        raise NeuralCheckpointError(
            "deprecated_bc_guide=true: refusing to treat BC checkpoint as search-move guide"
        )
    path = Path(str(ckpt))
    # Hard reject known BC construction checkpoints by filename.
    bad = {"bc_best_policy.pt", "bc_last_policy.pt"}
    if path.name in bad:
        raise NeuralCheckpointError(f"Refusing BC construction checkpoint as search guide: {path}")
    return TransformerSearchMoveGuide(config, checkpoint=path, device=device)
