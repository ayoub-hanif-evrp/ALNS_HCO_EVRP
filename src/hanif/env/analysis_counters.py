"""Runtime counters for duplicate expensive decision-analysis calls."""
from __future__ import annotations

import traceback
from collections import defaultdict

_COUNTERS: dict[str, int] = defaultdict(int)
_CYCLE_COUNTERS: dict[str, int] = defaultdict(int)
_DEBUG_WARN = False
_INSIDE_ANALYZE_DEPTH = 0

# Counters that should stay zero on the training hot path after analysis exists.
_HOTPATH_ZERO = (
    "dynamic_feature_recompute_calls",
    "station_usefulness_recompute_calls",
    "charge_portion_recompute_calls",
    "critic_forward_calls_actor_phase",
)


def set_debug_warn(enabled: bool) -> None:
    global _DEBUG_WARN
    _DEBUG_WARN = bool(enabled)


def enter_analyze() -> None:
    global _INSIDE_ANALYZE_DEPTH
    _INSIDE_ANALYZE_DEPTH += 1


def exit_analyze() -> None:
    global _INSIDE_ANALYZE_DEPTH
    _INSIDE_ANALYZE_DEPTH = max(0, _INSIDE_ANALYZE_DEPTH - 1)


def inside_analyze() -> bool:
    return _INSIDE_ANALYZE_DEPTH > 0


def bump(name: str, n: int = 1) -> None:
    _COUNTERS[name] += n
    _CYCLE_COUNTERS[name] += n
    if _DEBUG_WARN and name in _HOTPATH_ZERO and n > 0:
        stack = "".join(traceback.format_stack(limit=8)[:-1])
        print(
            f"[analysis_counters] WARNING: {name} += {n}\n{stack}",
            flush=True,
        )


def bump_outside_analyze(name: str, n: int = 1) -> None:
    """Bump only when not inside ``analyze_decision_state`` (duplicate detector)."""
    if _INSIDE_ANALYZE_DEPTH == 0:
        bump(name, n)


def get_counters() -> dict[str, int]:
    return dict(_COUNTERS)


def get_cycle_counters() -> dict[str, int]:
    return dict(_CYCLE_COUNTERS)


def reset_cycle_counters() -> None:
    _CYCLE_COUNTERS.clear()


def reset_all_counters() -> None:
    _COUNTERS.clear()
    _CYCLE_COUNTERS.clear()


def expected_per_decision_ok(cycle: dict[str, int] | None = None) -> list[str]:
    """Return warning strings for duplicate-call invariants (best-effort)."""
    c = cycle if cycle is not None else _CYCLE_COUNTERS
    warnings: list[str] = []
    for name in _HOTPATH_ZERO:
        if int(c.get(name, 0)) > 0:
            warnings.append(f"{name} > 0")
    return warnings


def snapshot_for_log(cycle: dict[str, int] | None = None) -> dict[str, int]:
    keys = (
        "analyze_decision_state_calls",
        "global_serviceability_calls",
        "controlled_closure_calls",
        "station_usefulness_recompute_calls",
        "charge_portion_recompute_calls",
        "dynamic_feature_recompute_calls",
        "feasible_customer_indices_calls",
        "recharge_search_calls",
        "critic_forward_calls_actor_phase",
    )
    c = cycle if cycle is not None else _CYCLE_COUNTERS
    return {k: int(c.get(k, 0)) for k in keys}
