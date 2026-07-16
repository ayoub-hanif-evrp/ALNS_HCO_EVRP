"""Deterministic C → R → RC instance interleaving (no heavy deps)."""
from __future__ import annotations


def build_interleaved_class_schedule(
    train_by_class: dict[str, list[str]],
    name_to_idx: dict[str, int],
    *,
    cycles: int,
) -> tuple[list[int], dict]:
    """Deterministic C→R→RC interleaving; identical order every cycle.

    Class lists are sorted by filename once. Smaller classes wrap with modulo so
    each cycle contains the same number of C, R, and RC steps
    (``3 * max(len_C, len_R, len_RC)``).
    """
    ordered = {
        "C": sorted(train_by_class.get("C") or []),
        "R": sorted(train_by_class.get("R") or []),
        "RC": sorted(train_by_class.get("RC") or []),
    }
    for cls, names in ordered.items():
        if not names:
            raise ValueError(f"Interleaved schedule requires non-empty train list for class {cls}")
        for name in names:
            if name not in name_to_idx:
                raise ValueError(f"Instance '{name}' missing from training dataset index")

    max_len = max(len(ordered["C"]), len(ordered["R"]), len(ordered["RC"]))
    one_cycle_names: list[str] = []
    for i in range(max_len):
        one_cycle_names.append(ordered["C"][i % len(ordered["C"])])
        one_cycle_names.append(ordered["R"][i % len(ordered["R"])])
        one_cycle_names.append(ordered["RC"][i % len(ordered["RC"])])

    one_cycle_idx = [name_to_idx[n] for n in one_cycle_names]
    schedule: list[int] = []
    for _ in range(max(int(cycles), 1)):
        schedule.extend(one_cycle_idx)

    meta = {
        "sampling": "interleaved_deterministic",
        "class_order": ["C", "R", "RC"],
        "sorted_train_by_class": {k: list(v) for k, v in ordered.items()},
        "updates_per_cycle": len(one_cycle_idx),
        "steps_per_class_per_cycle": max_len,
        "class_train_sizes": {k: len(v) for k, v in ordered.items()},
        "one_cycle_names": one_cycle_names,
        "cycles": int(cycles),
        "total_updates": len(schedule),
        "shuffle": False,
    }
    return schedule, meta
