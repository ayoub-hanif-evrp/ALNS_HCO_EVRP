"""No-regression gates for TG-FPS stage selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.search.score import SearchScore, compare_scores


@dataclass
class GateResult:
    accepted: bool
    winner: str
    challenger: str
    reason: str
    winner_score: SearchScore
    challenger_score: SearchScore
    details: dict = field(default_factory=dict)


def no_regression_gate(
    baseline_name: str,
    baseline_score: SearchScore,
    challenger_name: str,
    challenger_score: SearchScore,
) -> GateResult:
    """Accept challenger only if lexicographically not worse than baseline."""
    cmp = compare_scores(challenger_score, baseline_score)
    if cmp <= 0:
        return GateResult(
            accepted=True,
            winner=challenger_name if cmp < 0 else baseline_name,
            challenger=challenger_name,
            reason="improved_or_equal" if cmp < 0 else "equal",
            winner_score=challenger_score if cmp < 0 else baseline_score,
            challenger_score=challenger_score,
        )
    return GateResult(
        accepted=False,
        winner=baseline_name,
        challenger=challenger_name,
        reason="regression",
        winner_score=baseline_score,
        challenger_score=challenger_score,
    )


def neural_no_regression_gate(
    baseline_name: str,
    baseline_rows: list[dict],
    challenger_name: str,
    challenger_rows: list[dict],
    *,
    runtime_reduction_threshold: float = 0.15,
) -> GateResult:
    """Gate for neural vs deterministic on validation aggregates.

    Requires:
      - neural mean feasibility >= deterministic
      - neural mean served >= deterministic
    Then either lex-improve on used EVs / distance / duration / charging / J,
    or equal solution quality with meaningful runtime reduction.
    """
    def _agg(rows: list[dict]) -> dict:
        n = max(len(rows), 1)
        return {
            "feas": sum(1.0 if r.get("feasible") else 0.0 for r in rows) / n,
            "served": sum(float(r.get("served_ratio", 0)) for r in rows) / n,
            "used_evs": sum(float(r.get("used_evs", 0)) for r in rows) / n,
            "distance": sum(float(r.get("distance", 0)) for r in rows) / n,
            "duration": sum(float(r.get("duration", 0)) for r in rows) / n,
            "charging": sum(float(r.get("charging_time", 0)) for r in rows) / n,
            "J": sum(float(r.get("objective", 0)) for r in rows) / n,
            "runtime": sum(float(r.get("runtime", 0)) for r in rows) / n,
        }

    b = _agg(baseline_rows)
    c = _agg(challenger_rows)
    details = {"baseline": b, "challenger": c}

    # Aggregate SearchScore proxies for API compatibility.
    base_score = SearchScore(
        hard_violation_count=int(round((1.0 - b["feas"]) * 100)),
        unserved_customer_count=int(round((1.0 - b["served"]) * 100)),
        used_evs=int(round(b["used_evs"])),
        total_distance=b["distance"],
        total_charging_time=b["charging"],
        total_duration=b["duration"],
        paper_objective=b["J"],
    )
    chal_score = SearchScore(
        hard_violation_count=int(round((1.0 - c["feas"]) * 100)),
        unserved_customer_count=int(round((1.0 - c["served"]) * 100)),
        used_evs=int(round(c["used_evs"])),
        total_distance=c["distance"],
        total_charging_time=c["charging"],
        total_duration=c["duration"],
        paper_objective=c["J"],
    )

    if c["feas"] + 1e-12 < b["feas"]:
        return GateResult(
            False, baseline_name, challenger_name, "feasibility_regression",
            base_score, chal_score, details,
        )
    if c["served"] + 1e-12 < b["served"]:
        return GateResult(
            False, baseline_name, challenger_name, "served_regression",
            base_score, chal_score, details,
        )

    # Lex on remaining quality: used EVs → distance → charging → duration → J.
    quality_b = (b["used_evs"], b["distance"], b["charging"], b["duration"], b["J"])
    quality_c = (c["used_evs"], c["distance"], c["charging"], c["duration"], c["J"])
    if quality_c < quality_b:
        return GateResult(
            True, challenger_name, challenger_name, "quality_improved",
            chal_score, chal_score, details,
        )
    if quality_c == quality_b:
        if b["runtime"] > 0 and (b["runtime"] - c["runtime"]) / b["runtime"] >= runtime_reduction_threshold:
            return GateResult(
                True, challenger_name, challenger_name, "equal_quality_faster",
                chal_score, chal_score, details,
            )
        return GateResult(
            True, baseline_name, challenger_name, "equal_quality_no_speedup",
            base_score, chal_score, details,
        )
    # Quality worse despite feas/served OK → reject (unless somehow equal feas/served
    # but worse secondary — still regression).
    return GateResult(
        False, baseline_name, challenger_name, "quality_regression",
        base_score, chal_score, details,
    )


def select_best_method(candidates: dict[str, SearchScore]) -> tuple[str, SearchScore]:
    """Pick lexicographically best named method."""
    best_name = None
    best_score = None
    for name, score in candidates.items():
        if best_score is None or score.dominates(best_score):
            best_name, best_score = name, score
    assert best_name is not None and best_score is not None
    return best_name, best_score


def aggregate_instance_scores(scores: list[SearchScore]) -> SearchScore:
    """Mean aggregate for reporting (feas via hard/unserved rates encoded in counts)."""
    n = max(len(scores), 1)
    return SearchScore(
        hard_violation_count=int(round(sum(s.hard_violation_count for s in scores) / n)),
        unserved_customer_count=int(round(sum(s.unserved_customer_count for s in scores) / n)),
        used_evs=int(round(sum(s.used_evs for s in scores) / n)),
        total_distance=sum(s.total_distance for s in scores) / n,
        total_charging_time=sum(s.total_charging_time for s in scores) / n,
        total_duration=sum(s.total_duration for s in scores) / n,
        paper_objective=sum(s.paper_objective for s in scores) / n,
    )
