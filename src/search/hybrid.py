"""TG-FPS hybrid inference: deterministic Expert/Repair/ALNS + neural-guided variants."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.search.alns import run_alns
from src.search.neural_construction import run_neural_construction
from src.search.neural_guide import NeuralCheckpointError, NeuralSearchGuide
from src.search.portfolio import best_portfolio_solution
from src.search.repair import repair_solution
from src.search.route_replay import replay_solution
from src.search.score import SearchScore


METHOD_EXPERT = "Expert"
METHOD_EXPERT_REPAIR = "Expert+Repair"
METHOD_EXPERT_REPAIR_ALNS = "Expert+Repair+ALNS"
METHOD_NEURAL_REPAIR = "Neural-Guided Expert+Repair"
METHOD_NEURAL_REPAIR_ALNS = "Neural-Guided Expert+Repair+ALNS"

NEURAL_METHODS = {METHOD_NEURAL_REPAIR, METHOD_NEURAL_REPAIR_ALNS}
DETERMINISTIC_METHODS = {METHOD_EXPERT, METHOD_EXPERT_REPAIR, METHOD_EXPERT_REPAIR_ALNS}


@dataclass
class HybridSolveResult:
    method: str
    routes: list
    score: SearchScore
    m: int
    feasible: bool
    runtime_seconds: float
    stages: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    served_customers: int = 0
    n_customers: int = 0

    @property
    def served_ratio(self) -> float:
        return float(self.served_customers) / max(int(self.n_customers), 1)


def _finalize(instance, method, routes, score, m, feasible, t0, stages=None, telemetry=None):
    n_cust = int(instance.num_customers)
    unserved = int(score.unserved_customer_count) if score is not None else n_cust
    return HybridSolveResult(
        method=method,
        routes=routes,
        score=score,
        m=m,
        feasible=bool(feasible),
        runtime_seconds=time.perf_counter() - t0,
        stages=stages or {},
        telemetry=telemetry or {},
        served_customers=max(0, n_cust - unserved),
        n_customers=n_cust,
    )


def solve_expert(
    instance,
    m: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
) -> HybridSolveResult:
    """Deterministic expert portfolio construction (repair inside portfolio)."""
    t0 = time.perf_counter()
    cfg = config or {}
    w = weights or CostWeights()
    best = best_portfolio_solution(instance, m, config=cfg, weights=w, variants=["hybrid"], seeds=[0])
    final = replay_solution(instance, best.routes, config=cfg, weights=w, m=m)
    return _finalize(
        instance, METHOD_EXPERT, best.routes, final.score or best.score, m,
        final.feasible, t0,
    )


def solve_expert_repair(
    instance,
    m: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
) -> HybridSolveResult:
    t0 = time.perf_counter()
    cfg = config or {}
    w = weights or CostWeights()
    best = best_portfolio_solution(instance, m, config=cfg, weights=w, variants=["hybrid"], seeds=[0])
    routes, res = repair_solution(instance, best.routes, config=cfg, weights=w, m=m)
    final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
    return _finalize(
        instance, METHOD_EXPERT_REPAIR, routes, final.score or res.score or best.score, m,
        final.feasible, t0, stages={"repair": {"feasible": res.feasible}},
    )


def solve_expert_repair_alns(
    instance,
    m: int | None = None,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    with_alns: bool = True,
) -> HybridSolveResult:
    """Deterministic Expert + Repair (+ optional ALNS).

    If ``m`` is None, uses official ``solve_with_fleet_search`` (no fixed fleet).
    Passing an explicit ``m`` is allowed only for ablations / FixedFleet baselines.
    """
    if m is None:
        from src.search.fleet_search import solve_with_fleet_search
        from src.search.alns import run_alns as _alns

        t0 = time.perf_counter()
        cfg = config or {}
        w = weights or CostWeights()
        fleet = solve_with_fleet_search(instance, config=cfg, weights=w)
        routes, score, m_star = fleet.best_routes, fleet.best_score, fleet.m_star
        method = METHOD_EXPERT_REPAIR
        stages = {"fleet_search": {"m_star": m_star, "fleet_minimization_performed": True}}
        if with_alns:
            alns_cfg = (cfg.get("hybrid_search") or {}).get("alns") or {}
            alns = _alns(
                instance, routes, config=cfg, weights=w, m=m_star,
                iterations=int(alns_cfg.get("iterations", 40)),
                time_limit_seconds=float(alns_cfg.get("time_limit_seconds", 15.0)),
                guide=None,
            )
            routes, score = alns.routes, alns.score
            method = METHOD_EXPERT_REPAIR_ALNS
            stages["alns"] = {"iterations": alns.iterations, "accepted": alns.accepted}
        final = replay_solution(instance, routes, config=cfg, weights=w, m=m_star)
        return _finalize(
            instance, method, routes, final.score or score, m_star,
            final.feasible, t0, stages=stages,
            telemetry={"fleet_minimization_performed": True},
        )
    # Fixed-fleet ablation path (explicit m).
    if not with_alns:
        return solve_expert_repair(instance, m, config=config, weights=weights)
    t0 = time.perf_counter()
    cfg = config or {}
    hs = cfg.get("hybrid_search") or {}
    w = weights or CostWeights()
    best = best_portfolio_solution(instance, m, config=cfg, weights=w, variants=["hybrid"], seeds=[0])
    routes, res = repair_solution(instance, best.routes, config=cfg, weights=w, m=m)
    alns_cfg = hs.get("alns") or {}
    alns = run_alns(
        instance, routes, config=cfg, weights=w, m=m,
        iterations=int(alns_cfg.get("iterations", 40)),
        time_limit_seconds=float(alns_cfg.get("time_limit_seconds", 15.0)),
        guide=None,
    )
    final = replay_solution(instance, alns.routes, config=cfg, weights=w, m=m)
    return _finalize(
        instance, METHOD_EXPERT_REPAIR_ALNS, alns.routes, final.score or alns.score, m,
        final.feasible, t0,
        stages={"alns": {"iterations": alns.iterations, "accepted": alns.accepted},
                "fleet_minimization_performed": False, "fixed_fleet_m": m},
        telemetry={"fleet_minimization_performed": False, "fixed_fleet_m": m},
    )


def _require_guide(config: dict, guide: NeuralSearchGuide | None) -> NeuralSearchGuide:
    if guide is not None:
        return guide
    # Fail loudly — never silently degrade a neural method to expert-only.
    return NeuralSearchGuide(config)


def solve_neural_guided_expert_repair(
    instance,
    m: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    guide: NeuralSearchGuide | None = None,
    with_alns: bool = False,
) -> HybridSolveResult:
    """Neural-guided construction + exact repair (+ optional neural ALNS).

    Requires a loadable neural checkpoint. Raises ``NeuralCheckpointError`` if missing.
    """
    cfg = config or {}
    w = weights or (CostWeights.from_eval_config(cfg) if cfg else CostWeights())
    g = _require_guide(cfg, guide)
    g.reset_telemetry()
    t0 = time.perf_counter()

    # Only after successful load: advertise positive neural_variants in stages.
    hs = dict(cfg.get("hybrid_search") or {})
    cons = dict(hs.get("construction") or {})
    neural_variants = max(1, int(cons.get("neural_variants_requested", 1) or 1))
    cons["neural_variants"] = neural_variants
    hs["construction"] = cons

    routes, score, meta = run_neural_construction(
        instance, m, g, config=cfg, weights=w, repair=True,
    )
    method = METHOD_NEURAL_REPAIR
    stages: dict[str, Any] = {
        "construction": meta,
        "neural_variants": neural_variants,
        "checkpoint": g.telemetry.checkpoint_path,
        "checkpoint_sha256": g.telemetry.checkpoint_sha256,
    }

    if with_alns:
        alns_cfg = (hs.get("alns") or {})
        alns = run_alns(
            instance, routes, config=cfg, weights=w, m=m,
            iterations=int(alns_cfg.get("iterations", 40)),
            time_limit_seconds=float(alns_cfg.get("time_limit_seconds", 15.0)),
            guide=g,
        )
        routes, score = alns.routes, alns.score
        method = METHOD_NEURAL_REPAIR_ALNS
        stages["alns"] = {
            "iterations": alns.iterations,
            "accepted": alns.accepted,
            "telemetry": alns.telemetry,
        }

    final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
    if final.score is not None:
        score = final.score
    tel = g.telemetry.as_dict()
    if not tel.get("neural_participation"):
        raise NeuralCheckpointError(
            f"{method}: checkpoint loaded but made zero neural decisions/scores — "
            "refusing to label this run as neural-guided."
        )
    return _finalize(
        instance, method, routes, score, m, final.feasible, t0,
        stages=stages, telemetry=tel,
    )


def solve_method(
    method: str,
    instance,
    m: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    guide: NeuralSearchGuide | None = None,
) -> HybridSolveResult:
    """Dispatch explicit method names."""
    if method == METHOD_EXPERT:
        return solve_expert(instance, m, config=config, weights=weights)
    if method == METHOD_EXPERT_REPAIR:
        return solve_expert_repair(instance, m, config=config, weights=weights)
    if method == METHOD_EXPERT_REPAIR_ALNS:
        return solve_expert_repair_alns(instance, m, config=config, weights=weights, with_alns=True)
    if method == METHOD_NEURAL_REPAIR:
        return solve_neural_guided_expert_repair(
            instance, m, config=config, weights=weights, guide=guide, with_alns=False,
        )
    if method == METHOD_NEURAL_REPAIR_ALNS:
        return solve_neural_guided_expert_repair(
            instance, m, config=config, weights=weights, guide=guide, with_alns=True,
        )
    raise ValueError(f"Unknown method: {method}")


# Backward-compatible aliases (deterministic only — not neural TG-FPS).
def solve_tgfps(instance, **kwargs):
    """Deprecated alias: deterministic portfolio+ALNS, not neural-guided."""
    m = kwargs.pop("m", None) or 20
    use_alns = kwargs.pop("use_alns", True)
    kwargs.pop("use_portfolio", None)
    kwargs.pop("time_limit_seconds", None)
    if use_alns:
        return solve_expert_repair_alns(instance, m, with_alns=True, **kwargs)
    return solve_expert_repair(instance, m, **kwargs)
