"""Deterministic C → R → RC class-balanced macro-update schedules."""
from __future__ import annotations


def build_interleaved_class_schedule(
    train_by_class: dict[str, list[str]],
    name_to_idx: dict[str, int],
    *,
    cycles: int,
    macro_updates: bool = True,
    type2_warmup_cycles: int = 0,
) -> tuple[list, dict]:
    """Deterministic C→R→RC schedule; identical order every cycle.

    Class lists are sorted by filename once. Smaller classes wrap with modulo so
    each cycle contains the same number of C, R, and RC steps.

    When ``macro_updates`` is True (default), each schedule entry is a triplet
    ``[c_idx, r_idx, rc_idx]`` forming one optimizer update. When False, the
    schedule is a flat list of single indices (legacy per-instance updates).

    If ``type2_warmup_cycles`` > 0, the first that many cycles only sample type-2
    (*2xx) instances within each class (curriculum warm-start).
    """
    from src.schneider.data.mixed_class_split import _is_type2_instance

    ordered_all = {
        "C": sorted(train_by_class.get("C") or []),
        "R": sorted(train_by_class.get("R") or []),
        "RC": sorted(train_by_class.get("RC") or []),
    }
    ordered_t2 = {
        cls: [n for n in names if _is_type2_instance(n)]
        for cls, names in ordered_all.items()
    }
    for cls, names in ordered_all.items():
        if not names:
            raise ValueError(f"Interleaved schedule requires non-empty train list for class {cls}")
        for name in names:
            if name not in name_to_idx:
                raise ValueError(f"Instance '{name}' missing from training dataset index")

    def _one_cycle(ordered: dict[str, list[str]]) -> tuple[list[list[int]] | list[int], list[str], list[tuple[str, str, str]], int]:
        # Fall back to all-class lists if a class has no type-2 members.
        use = {
            cls: (ordered[cls] if ordered[cls] else ordered_all[cls])
            for cls in ("C", "R", "RC")
        }
        max_len = max(len(use["C"]), len(use["R"]), len(use["RC"]))
        triples: list[tuple[str, str, str]] = []
        for i in range(max_len):
            triples.append((
                use["C"][i % len(use["C"])],
                use["R"][i % len(use["R"])],
                use["RC"][i % len(use["RC"])],
            ))
        names: list[str] = []
        for c_name, r_name, rc_name in triples:
            names.extend([c_name, r_name, rc_name])
        if macro_updates:
            batches = [
                [name_to_idx[c], name_to_idx[r], name_to_idx[rc]]
                for c, r, rc in triples
            ]
            return batches, names, triples, len(batches)
        idxs = [name_to_idx[n] for n in names]
        return idxs, names, triples, len(idxs)

    warmup = max(0, int(type2_warmup_cycles or 0))
    n_cycles = max(int(cycles), 1)
    schedule: list = []
    # Use full-pool cycle length as the canonical updates_per_cycle for logging.
    full_batches, full_names, full_triples, updates_per_cycle = _one_cycle(ordered_all)
    for cyc in range(n_cycles):
        if cyc < warmup and all(ordered_t2[cls] for cls in ("C", "R", "RC")):
            batches, _, _, _ = _one_cycle(ordered_t2)
        else:
            batches = full_batches
        schedule.extend(batches)

    meta = {
        "sampling": "macro_class_balanced" if macro_updates else "interleaved_deterministic",
        "macro_updates": bool(macro_updates),
        "class_order": ["C", "R", "RC"],
        "sorted_train_by_class": {k: list(v) for k, v in ordered_all.items()},
        "updates_per_cycle": updates_per_cycle,
        "steps_per_class_per_cycle": max(
            len(ordered_all["C"]), len(ordered_all["R"]), len(ordered_all["RC"])
        ),
        "class_train_sizes": {k: len(v) for k, v in ordered_all.items()},
        "one_cycle_names": full_names,
        "one_cycle_triples": [
            {"C": c, "R": r, "RC": rc} for c, r, rc in full_triples
        ],
        "cycles": n_cycles,
        "total_updates": len(schedule),
        "shuffle": False,
        "type2_warmup_cycles": warmup,
    }
    return schedule, meta
