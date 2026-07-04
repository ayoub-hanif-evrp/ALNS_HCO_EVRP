"""Generate live random-instance solution diagnostics during training."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.evrptw_dataset import split_dataset
from src.data.rc21_split import resolve_rc21_split
from src.env.cost import CostWeights
from src.training.flat_schedule import sample_fleet_size
from src.training.rollout_trace import run_traced_episode
from src.training.solution_diagnostics import LIVE_DIAGNOSTIC_GROUPS, generate_live_diagnostic_figures
from src.training.train_rl import load_policy
from src.utils.logging_utils import get_logger

LOGGER = get_logger()
_STATE_FILE = "live_state.json"
_LOCK = threading.Lock()
_GENERATING = False


def dashboard_live_cfg(cfg: dict) -> dict:
    dcfg = dict(cfg.get("dashboard") or {})
    dcfg.setdefault("live_random_diagnostics", True)
    dcfg.setdefault("live_random_interval_sec", 45)
    dcfg.setdefault("live_refresh_min_epoch_delta", 1)
    dcfg.setdefault("trace_mode", "greedy")
    return dcfg


def resolve_checkpoint(cfg: dict) -> Path | None:
    ckpt_dir = Path((cfg.get("outputs") or {}).get("checkpoints", "outputs/checkpoints"))
    if not ckpt_dir.is_dir():
        return None
    for name in ("last.pt", "best.pt", "best_cycle_objective.pt"):
        path = ckpt_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return epoch_ckpts[0] if epoch_ckpts else None


def load_training_dataset(cfg: dict):
    train_names, eval_names, _ = resolve_rc21_split(cfg)
    ds_cfg = dict(cfg.get("dataset") or {})
    ds_cfg["train_instances"] = train_names
    ds_cfg["eval_instances"] = eval_names
    seed = int(cfg.get("seed", 42))
    train_ds, _ = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter"),
        eval_filter=ds_cfg.get("eval_size_filter"),
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
    )
    return train_ds


def _state_path(figures_dir: Path) -> Path:
    return figures_dir / "live" / _STATE_FILE


def read_live_state(figures_dir: Path) -> dict[str, Any]:
    path = _state_path(figures_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_live_state(figures_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(figures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_refresh_live(cfg: dict, df: pd.DataFrame, state: dict[str, Any]) -> bool:
    dcfg = dashboard_live_cfg(cfg)
    if not dcfg.get("live_random_diagnostics", True):
        return False
    if not state:
        return True
    if state.get("generating"):
        return False
    interval = float(dcfg.get("live_random_interval_sec", 45))
    if time.time() - float(state.get("generated_at", 0)) >= interval:
        return True
    if not df.empty:
        cur_epoch = int(df.iloc[-1].get("epoch", 0))
        min_delta = int(dcfg.get("live_refresh_min_epoch_delta", 1))
        if cur_epoch - int(state.get("epoch", -1)) >= min_delta:
            return True
    ckpt = resolve_checkpoint(cfg)
    if ckpt and ckpt.stat().st_mtime > float(state.get("generated_at", 0)):
        return True
    return False


def _enrich_live_item(figures_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = figures_dir / item["file"]
    exists = path.is_file() and path.stat().st_size > 0
    return {**item, "exists": exists, "mtime": int(path.stat().st_mtime) if exists else 0}


def build_live_groups(figures_dir: Path) -> list[dict[str, Any]]:
    groups = []
    for group in LIVE_DIAGNOSTIC_GROUPS:
        items = [_enrich_live_item(figures_dir, item) for item in group["items"]]
        groups.append({**group, "items": items})
    return groups


def _generate_live_snapshot(cfg: dict, df: pd.DataFrame, figures_dir: Path, traces_dir: Path) -> dict[str, Any]:
    ckpt = resolve_checkpoint(cfg)
    if ckpt is None:
        return {"error": "No checkpoint yet — waiting for first policy save (best.pt or last.pt)."}

    dcfg = dashboard_live_cfg(cfg)
    prob = cfg.get("problem") or {}
    decode = cfg.get("decoding") or {}
    hard_tw = str((cfg.get("constraints") or {}).get("time_windows_mode", "")) == "hard_original"
    device = "cpu"
    policy = load_policy(ckpt, device=device)
    train_ds = load_training_dataset(cfg)

    bucket = int(time.time() // max(1.0, float(dcfg.get("live_random_interval_sec", 45))))
    rng = random.Random(int(cfg.get("seed", 42)) + bucket)
    inst_idx = rng.randrange(len(train_ds))
    instance = train_ds[inst_idx]
    log_row = df.iloc[-1].to_dict() if not df.empty else {}
    fleet_rng = random.Random(bucket + inst_idx)
    m = sample_fleet_size(instance, 3, fleet_rng)
    weights = CostWeights.from_config(cfg["cost"])

    _, trace = run_traced_episode(
        instance,
        policy,
        m,
        weights,
        mode=str(dcfg.get("trace_mode", decode.get("mode", "greedy"))),
        device=device,
        charging_portions=policy.charging_portions,
        use_horizon_check=bool(prob.get("use_horizon_check", True)),
        use_depot_charging=bool(prob.get("use_depot_charging", True)),
        max_steps_factor=int(decode.get("max_steps_factor", 6)),
        hard_time_windows=hard_tw,
        selection_tag="live_random",
        log_row={
            **log_row,
            "instance_name": getattr(instance, "instance_name", train_ds.names[inst_idx]),
            "instance_idx": inst_idx,
        },
    )

    live_dir = figures_dir / "live"
    generate_live_diagnostic_figures(trace, live_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    (traces_dir / "trace_live_random.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8",
    )

    meta = trace.get("meta") or {}
    return {
        "generated_at": time.time(),
        "generating": False,
        "epoch": int(log_row.get("epoch", meta.get("epoch", 0))),
        "cycle": int(log_row.get("cycle", meta.get("cycle", 0))),
        "instance_name": meta.get("instance_name", train_ds.names[inst_idx]),
        "instance_idx": inst_idx,
        "checkpoint": ckpt.name,
        "objective": float(meta.get("objective", 0)),
        "served_ratio": float(meta.get("served_ratio", 0)),
        "feasible": bool(meta.get("feasible", False)),
        "unserved_customers": int(meta.get("unserved_customers", 0)),
        "used_evs": int(meta.get("used_evs", 0)),
        "error": None,
    }


def _run_generation_worker(cfg: dict, df: pd.DataFrame, figures_dir: Path, traces_dir: Path) -> None:
    global _GENERATING
    state = read_live_state(figures_dir)
    state["generating"] = True
    state["started_at"] = time.time()
    write_live_state(figures_dir, state)
    try:
        new_state = _generate_live_snapshot(cfg, df, figures_dir, traces_dir)
        write_live_state(figures_dir, new_state)
        LOGGER.info(
            "Live diagnostics: %s @ epoch %s (checkpoint %s)",
            new_state.get("instance_name"),
            new_state.get("epoch"),
            new_state.get("checkpoint"),
        )
    except Exception as exc:
        LOGGER.warning("Live random diagnostics failed: %s", exc)
        state = read_live_state(figures_dir)
        state.update({"generating": False, "error": str(exc), "generated_at": time.time()})
        write_live_state(figures_dir, state)
    finally:
        _GENERATING = False


def ensure_live_random_diagnostics(
    cfg: dict,
    df: pd.DataFrame,
    figures_dir: Path,
    traces_dir: Path,
) -> dict[str, Any]:
    """Return live snapshot payload; kick off background generation when stale."""
    global _GENERATING
    dcfg = dashboard_live_cfg(cfg)
    if not dcfg.get("live_random_diagnostics", True):
        return {"enabled": False}

    figures_dir = Path(figures_dir)
    state = read_live_state(figures_dir)

    if should_refresh_live(cfg, df, state) and not _GENERATING:
        if _LOCK.acquire(blocking=False):
            try:
                if not _GENERATING:
                    _GENERATING = True
                    threading.Thread(
                        target=_run_generation_worker,
                        args=(cfg, df, figures_dir, traces_dir),
                        daemon=True,
                    ).start()
            finally:
                _LOCK.release()

    state = read_live_state(figures_dir)
    groups = build_live_groups(figures_dir)
    any_figure = any(item["exists"] for group in groups for item in group["items"])
    age_sec = None
    if state.get("generated_at"):
        age_sec = max(0, int(time.time() - float(state["generated_at"])))

    return {
        "enabled": True,
        "groups": groups,
        "any_figure": any_figure,
        "generating": bool(state.get("generating") or _GENERATING),
        "state": state,
        "age_sec": age_sec,
        "interval_sec": int(dcfg.get("live_random_interval_sec", 45)),
        "error": state.get("error"),
    }
