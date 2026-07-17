"""Deterministic C → R → RC class-balanced macro-update schedules."""
from __future__ import annotations


def build_interleaved_class_schedule(
    train_by_class: dict[str, list[str]],
    name_to_idx: dict[str, int],
    *,
    cycles: int,
    macro_updates: bool = True,
) -> tuple[list, dict]:
    """Deterministic C→R→RC schedule; identical order every cycle.

    Class lists are sorted by filename once. Smaller classes wrap with modulo so
    each cycle contains the same number of C, R, and RC steps.

    When ``macro_updates`` is True (default), each schedule entry is a triplet
    ``[c_idx, r_idx, rc_idx]`` forming one optimizer update. When False, the
    schedule is a flat list of single indices (legacy per-instance updates).
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
    one_cycle_triples: list[tuple[str, str, str]] = []
    for i in range(max_len):
        one_cycle_triples.append((
            ordered["C"][i % len(ordered["C"])],
            ordered["R"][i % len(ordered["R"])],
            ordered["RC"][i % len(ordered["RC"])],
        ))

    one_cycle_names: list[str] = []
    for c_name, r_name, rc_name in one_cycle_triples:
        one_cycle_names.extend([c_name, r_name, rc_name])

    if macro_updates:
        one_cycle_batches = [
            [name_to_idx[c], name_to_idx[r], name_to_idx[rc]]
            for c, r, rc in one_cycle_triples
        ]
        schedule: list = []
        for _ in range(max(int(cycles), 1)):
            schedule.extend(one_cycle_batches)
        updates_per_cycle = len(one_cycle_batches)
        sampling = "macro_class_balanced"
    else:
        one_cycle_idx = [name_to_idx[n] for n in one_cycle_names]
        schedule = []
        for _ in range(max(int(cycles), 1)):
            schedule.extend(one_cycle_idx)
        updates_per_cycle = len(one_cycle_idx)
        sampling = "interleaved_deterministic"

    meta = {
        "sampling": sampling,
        "macro_updates": bool(macro_updates),
        "class_order": ["C", "R", "RC"],
        "sorted_train_by_class": {k: list(v) for k, v in ordered.items()},
        "updates_per_cycle": updates_per_cycle,
        "steps_per_class_per_cycle": max_len,
        "class_train_sizes": {k: len(v) for k, v in ordered.items()},
        "one_cycle_names": one_cycle_names,
        "one_cycle_triples": [
            {"C": c, "R": r, "RC": rc} for c, r, rc in one_cycle_triples
        ],
        "cycles": int(cycles),
        "total_updates": len(schedule),
        "shuffle": False,
    }
    return schedule, meta
