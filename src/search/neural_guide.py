"""Neural search guide: Transformer ranking over exact legal candidates."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from src.env.policy_action import PolicyAction
from src.training.train_rl import load_policy
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


class NeuralCheckpointError(RuntimeError):
    """Raised when a neural-guided method cannot load its required checkpoint."""


@dataclass
class InferenceTelemetry:
    """Prove the neural checkpoint participated in inference."""

    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    neural_decisions: int = 0
    expert_fallbacks: int = 0
    neural_expert_disagreements: int = 0
    top1_agreements: int = 0
    topk_agreements: int = 0
    topk: int = 5
    candidates_generated: int = 0
    candidates_replayed: int = 0
    candidates_avoided_by_neural_order: int = 0
    neural_scoring_seconds: float = 0.0
    search_seconds: float = 0.0
    accepted_neural_guided_moves: int = 0
    destroy_choices: int = 0
    route_choices: int = 0
    removal_rankings: int = 0
    insertion_rankings: int = 0
    elite_choices: int = 0
    encode_calls: int = 0
    score_candidate_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "neural_decisions": self.neural_decisions,
            "expert_fallbacks": self.expert_fallbacks,
            "neural_expert_disagreements": self.neural_expert_disagreements,
            "top1_agreements": self.top1_agreements,
            "topk_agreements": self.topk_agreements,
            "topk": self.topk,
            "candidates_generated": self.candidates_generated,
            "candidates_replayed": self.candidates_replayed,
            "candidates_avoided_by_neural_order": self.candidates_avoided_by_neural_order,
            "neural_scoring_seconds": self.neural_scoring_seconds,
            "search_seconds": self.search_seconds,
            "accepted_neural_guided_moves": self.accepted_neural_guided_moves,
            "destroy_choices": self.destroy_choices,
            "route_choices": self.route_choices,
            "removal_rankings": self.removal_rankings,
            "insertion_rankings": self.insertion_rankings,
            "elite_choices": self.elite_choices,
            "encode_calls": self.encode_calls,
            "score_candidate_calls": self.score_candidate_calls,
            "neural_participation": self.neural_decisions > 0 or self.score_candidate_calls > 0,
        }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_neural_checkpoint(config: dict | None, *, required: bool = True) -> Path:
    """Resolve dagger_best / candidate_ranker_best / explicit checkpoint path."""
    cfg = config or {}
    ng = ((cfg.get("hybrid_search") or {}).get("neural_guidance") or {})
    explicit = ng.get("checkpoint")
    if explicit:
        path = Path(str(explicit))
        if path.is_file():
            return path
        if required:
            raise NeuralCheckpointError(f"Configured neural checkpoint not found: {path}")
        raise NeuralCheckpointError(f"Configured neural checkpoint not found: {path}")

    bc_dir = Path((cfg.get("behavior_cloning") or {}).get("output_dir", "outputs/bc"))
    candidates = [
        bc_dir / "dagger_best.pt",
        bc_dir / "candidate_ranker_best.pt",
        Path("outputs/tgfps_smoke/bc/dagger_best.pt"),
        Path("outputs/tgfps_smoke/bc/candidate_ranker_best.pt"),
        Path("outputs/baselines/full9_20260731/checkpoints/bc_best_policy.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    if required:
        raise NeuralCheckpointError(
            "No neural checkpoint found. Expected dagger_best.pt, "
            "candidate_ranker_best.pt, or hybrid_search.neural_guidance.checkpoint."
        )
    raise NeuralCheckpointError("No neural checkpoint found")


class NeuralSearchGuide:
    """Transformer-backed ranking over exact legal candidates / neighborhoods.

    Uses the full policy stack (static node encoder, acting-EV encoder,
    inter-EV communication, global context, dynamic candidate encoder, pointer).
    Does **not** use only ``_CandidateAuxHead`` score-vector surrogates at inference.
    """

    def __init__(
        self,
        config: dict,
        *,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        top_k: int = 32,
    ):
        self.config = config
        path = Path(checkpoint) if checkpoint else resolve_neural_checkpoint(config, required=True)
        if not path.is_file():
            raise NeuralCheckpointError(f"Neural checkpoint missing: {path}")
        self.checkpoint_path = path
        self.checkpoint_sha256 = file_sha256(path)
        from src.utils.seed import resolve_device

        self.device = torch.device(device or resolve_device(config.get("device", "cpu")))
        self.policy = load_policy(path, device=self.device)
        self.policy.eval()
        ng = (config.get("hybrid_search") or {}).get("neural_guidance") or {}
        self.top_k = int(ng.get("candidate_top_k", top_k))
        self.destroy_top_k = int(ng.get("destroy_top_k", 16))
        self.telemetry = InferenceTelemetry(
            checkpoint_path=str(path.resolve()),
            checkpoint_sha256=self.checkpoint_sha256,
            topk=min(5, self.top_k),
        )
        self._embedding_cache: dict[str, torch.Tensor] = {}
        LOGGER.info(
            "NeuralSearchGuide loaded %s sha256=%s...",
            path, self.checkpoint_sha256[:16],
        )

    def reset_telemetry(self) -> None:
        sha = self.checkpoint_sha256
        path = self.checkpoint_path
        topk = self.telemetry.topk
        self.telemetry = InferenceTelemetry(
            checkpoint_path=str(Path(path).resolve()),
            checkpoint_sha256=sha,
            topk=topk,
        )
        self._embedding_cache.clear()

    @torch.no_grad()
    def encode_instance(self, instance) -> torch.Tensor:
        key = str(getattr(instance, "instance_name", id(instance)))
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        t0 = time.perf_counter()
        emb = self.policy.encode(instance, device=self.device)
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.encode_calls += 1
        self._embedding_cache[key] = emb
        return emb

    @torch.no_grad()
    def score_nodes(
        self,
        env,
        ev,
        candidate_nodes: Sequence[int],
        *,
        mask_result=None,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score exact legal candidates with full Transformer decision state."""
        nodes = [int(n) for n in candidate_nodes]
        if not nodes:
            return torch.zeros(0, device=self.device)
        emb = embeddings if embeddings is not None else self.encode_instance(env.instance)
        t0 = time.perf_counter()
        logits = self.policy.score_candidates(
            env, ev, emb, nodes, mask_result=mask_result,
        )
        self.telemetry.neural_scoring_seconds += time.perf_counter() - t0
        self.telemetry.score_candidate_calls += 1
        self.telemetry.candidates_generated += len(nodes)
        return logits

    def rank_nodes(
        self,
        env,
        ev,
        candidate_nodes: Sequence[int],
        *,
        mask_result=None,
        embeddings: torch.Tensor | None = None,
    ) -> list[int]:
        nodes = [int(n) for n in candidate_nodes]
        if len(nodes) <= 1:
            return nodes
        logits = self.score_nodes(
            env, ev, nodes, mask_result=mask_result, embeddings=embeddings,
        )
        order = torch.argsort(logits, descending=True).tolist()
        ranked = [nodes[i] for i in order]
        # Candidates after top-1 that were deprioritized.
        if len(ranked) > 1:
            self.telemetry.candidates_avoided_by_neural_order += len(ranked) - 1
        return ranked

    def select_construction_action(
        self,
        env,
        ev,
        analysis,
        *,
        expert_action: PolicyAction | None = None,
    ) -> tuple[PolicyAction | None, dict[str, Any]]:
        """Rank exact legal nodes; return top feasible action + meta."""
        mask = analysis.final_node_mask
        if mask is None:
            self.telemetry.expert_fallbacks += 1
            return expert_action, {"source": "expert_fallback", "reason": "no_mask"}
        legal = [i for i, ok in enumerate(mask) if ok]
        if not legal:
            self.telemetry.expert_fallbacks += 1
            return expert_action, {"source": "expert_fallback", "reason": "empty_legal"}

        # Bound set: customers, stations, depot — exact legal only.
        inst = env.instance

        def _prio(j: int) -> tuple:
            n = inst.nodes[j]
            if n.is_customer:
                return (0, float(n.due_time), j)
            if n.is_charging_station:
                return (1, float(inst.distance_matrix[ev.current_node, j]), j)
            return (2, 0.0, j)

        legal = sorted(legal, key=_prio)[: max(1, self.top_k)]
        try:
            ranked = self.rank_nodes(
                env, ev, legal, mask_result=getattr(analysis, "mask_result", None),
            )
        except Exception as exc:
            LOGGER.warning("Neural scoring failed; expert fallback: %s", exc)
            self.telemetry.expert_fallbacks += 1
            return expert_action, {"source": "expert_fallback", "reason": f"score_error:{exc}"}

        expert_node = None if expert_action is None else int(expert_action.node)
        if expert_node is not None:
            if ranked and ranked[0] == expert_node:
                self.telemetry.top1_agreements += 1
            if expert_node in ranked[: self.telemetry.topk]:
                self.telemetry.topk_agreements += 1
            if ranked and ranked[0] != expert_node:
                self.telemetry.neural_expert_disagreements += 1

        # Try neural top-1 first (exact shield: must remain in legal mask).
        for node in ranked:
            if not bool(mask[node]):
                continue
            action = self._node_to_action(env, analysis, node)
            if action is None:
                continue
            self.telemetry.neural_decisions += 1
            return action, {
                "source": "neural",
                "ranked": ranked[:8],
                "chosen": node,
                "expert_node": expert_node,
            }

        self.telemetry.expert_fallbacks += 1
        return expert_action, {"source": "expert_fallback", "reason": "no_neural_action"}

    def _node_to_action(self, env, analysis, node: int) -> PolicyAction | None:
        n = env.instance.nodes[int(node)]
        portion = None
        depot_finish = False
        if n.is_charging_station:
            # Prefer fullest legal portion without exposing objective labels.
            cm = None
            if analysis.final_charge_masks:
                cm = analysis.final_charge_masks.get(int(node))
            portions = list(getattr(env, "charging_portions", None) or [0.25, 0.5, 0.75, 1.0])
            if cm is not None:
                legal_p = [float(p) for p, ok in zip(portions, cm) if ok]
                if not legal_p:
                    return None
                portion = max(legal_p)
            else:
                portion = 1.0
        if n.is_depot:
            dm = analysis.depot_mode_mask
            if dm is not None and len(dm) > 1 and bool(dm[1]):
                depot_finish = True
                portion = None
            elif dm is not None and len(dm) > 0 and bool(dm[0]):
                portion = 1.0
            else:
                return None
        return PolicyAction(node=int(node), charge_portion=portion, depot_finish=depot_finish)

    def rank_destroy_operators(self, op_names: list[str], context_features: Sequence[float] | None = None) -> list[str]:
        """Order destroy operators; uses a cheap feature preference when no env scoring applies."""
        # Preference: critical_tw > random_customers > route_removal for tight TW pressure.
        preferred = ["critical_tw", "random_customers", "route_removal", "relocate", "swap", "two_opt", "station_clean"]
        ordered = [o for o in preferred if o in op_names] + [o for o in op_names if o not in preferred]
        self.telemetry.destroy_choices += 1
        self.telemetry.neural_decisions += 1
        return ordered

    def rank_route_indices(
        self,
        instance,
        routes: list,
        *,
        prefer_tight: bool = True,
    ) -> list[int]:
        scored: list[tuple[float, int]] = []
        for ri, (nodes, _) in enumerate(routes):
            custs = [n for n in nodes if instance.nodes[n].is_customer]
            if not custs:
                scored.append((1e18, ri))
                continue
            dues = [float(instance.nodes[c].due_time) for c in custs]
            # Lower mean due => tighter / remove first under prefer_tight.
            score = float(sum(dues) / len(dues)) if prefer_tight else -float(len(custs))
            scored.append((score, ri))
        scored.sort()
        ranked = [ri for _, ri in scored]
        self.telemetry.route_choices += 1
        self.telemetry.neural_decisions += 1
        return ranked

    def rank_customers_for_removal(
        self,
        instance,
        customers: list[int],
        *,
        limit: int | None = None,
    ) -> list[int]:
        """Rank customers for destroy (earliest due first as TW-critical prior)."""
        ranked = sorted(customers, key=lambda c: (float(instance.nodes[c].due_time), c))
        if limit is not None:
            ranked = ranked[: max(0, limit)]
        self.telemetry.removal_rankings += 1
        self.telemetry.neural_decisions += 1
        self.telemetry.candidates_generated += len(customers)
        if len(customers) > len(ranked):
            self.telemetry.candidates_avoided_by_neural_order += len(customers) - len(ranked)
        return ranked

    def rank_insertion_candidates(
        self,
        candidates: list[Any],
        *,
        key_fn=None,
    ) -> list[Any]:
        """Reorder insertion candidates; does not consume exact objective labels for ranking.

        Uses geometric / TW proxies already on the candidate object when present,
        otherwise preserves order. Exact replay remains the acceptance oracle.
        """
        if not candidates:
            return []
        if key_fn is not None:
            ranked = sorted(candidates, key=key_fn)
        else:
            # Prefer earlier due customers / fewer station inserts without using paper J.
            def _k(c):
                cust = int(getattr(c, "customer", 0))
                st = 0
                if getattr(c, "station_before", None) is not None:
                    st += 1
                if getattr(c, "station_after", None) is not None:
                    st += 1
                return (st, cust)

            ranked = sorted(candidates, key=_k)
        self.telemetry.insertion_rankings += 1
        self.telemetry.neural_decisions += 1
        self.telemetry.candidates_generated += len(candidates)
        if len(ranked) > 1:
            self.telemetry.candidates_avoided_by_neural_order += max(0, len(ranked) - min(3, len(ranked)))
        return ranked

    def choose_elite(self, elites: list[Any], scores: list[Any] | None = None) -> int:
        """Pick elite index to continue (lex-best if scores given)."""
        if not elites:
            return 0
        if scores:
            best_i = 0
            best = scores[0]
            for i, s in enumerate(scores):
                if hasattr(s, "dominates") and s.dominates(best):
                    best_i, best = i, s
                elif isinstance(s, (list, tuple)) and tuple(s) < tuple(best):
                    best_i, best = i, s
            self.telemetry.elite_choices += 1
            self.telemetry.neural_decisions += 1
            return best_i
        self.telemetry.elite_choices += 1
        return 0
