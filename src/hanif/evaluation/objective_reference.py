"""Φ normalizers from the frozen Classical solution on the same instance."""
from __future__ import annotations

import json
from pathlib import Path

from src.hanif.evaluation.paper_objective import MODE_CLASSICAL, ObjectiveReference, objective_spec, phi_from_replay
from src.hanif.search.route_replay import replay_solution


def reference_from_classical_routes(instance, routes, config: dict | None = None, weights=None) -> ObjectiveReference:
    """Measure D / Tstation / Ccharge of a finished Classical solution at ρ=0."""
    cfg = dict(config or {})
    replay = replay_solution(instance, routes, config=cfg, weights=weights, m=max(len(routes), 1))
    measure = dict(cfg)
    measure["paper_objective"] = {
        "mode": "expected_heterogeneous",
        "risk_blend": 0.0,
        "risk_alpha": 0.95,
    }
    info = phi_from_replay(instance, replay, measure)
    return ObjectiveReference(
        distance=max(float(info["distance"]), 0.0),
        tstation=max(float(info["tstation"]), 0.0),
        ccharge=max(float(info.get("expected_charging_cost") or info["crisk"]), 0.0),
    )


def reference_path(root: Path, variant: str, name: str) -> Path:
    key = "original_tw" if variant in ("A", "original_tw") else "relaxed_tw"
    return Path(root) / key / f"{name}.json"


def save_reference(path: Path, ref: ObjectiveReference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref.as_dict(), indent=2) + "\n", encoding="utf-8")


def load_reference(path: Path) -> ObjectiveReference | None:
    if not path.is_file():
        return None
    return ObjectiveReference.from_dict(json.loads(path.read_text(encoding="utf-8")))


def attach_reference(instance, config: dict, weights, *, reference_dir: Path | None = None,
                     variant: str | None = None) -> dict:
    """Classical search needs no Φ reference. Proposed search requires a frozen Classical file."""
    cfg = dict(config)
    spec = objective_spec(cfg)
    if spec["mode"] == MODE_CLASSICAL:
        return cfg
    existing = cfg.get("objective_reference")
    if existing:
        return cfg
    name = getattr(instance, "instance_name", None)
    if reference_dir is not None and variant and name:
        loaded = load_reference(reference_path(reference_dir, variant, str(name)))
        if loaded is not None:
            cfg["objective_reference"] = loaded.as_dict()
            return cfg
    raise RuntimeError(
        "Proposed Expected Heterogeneous search requires a frozen Classical "
        "ObjectiveReference (D_class, Tstation_class, Ccharge_class). "
        "Run Classical first and pass --reference-dir."
    )


def build_objective_reference(*_args, **_kwargs):
    raise RuntimeError(
        "dense_construct is not the paper Φ normalizer. "
        "Use reference_from_classical_routes on a finished Classical solution."
    )
