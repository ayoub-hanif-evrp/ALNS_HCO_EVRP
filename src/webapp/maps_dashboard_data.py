"""Data layer for the maps / solution-behavior explorer (no NiceGUI)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.evrptw_dataset import split_dataset
from src.data.rc21_split import resolve_rc21_split
from src.env.cost import CostWeights
from src.training.flat_schedule import sample_fleet_size
from src.training.rollout_trace import run_traced_episode
from src.training.solution_diagnostics import LIVE_PLOT_SPECS, generate_live_diagnostic_figures
from src.training.train_rl import load_checkpoint_training_meta, load_policy
from src.utils.logging_utils import get_logger

LOGGER = get_logger()
MAPS_MOUNT = "/maps-figures"


def maps_cfg(cfg: dict) -> dict:
    mcfg = dict(cfg.get("maps_dashboard") or {})
    mcfg.setdefault("cache_snapshots", True)
    mcfg.setdefault("trace_mode", "greedy")
    mcfg.setdefault("use_cached_if_present", True)
    return mcfg


def maps_figures_root(cfg: dict, outputs_root: Path | None = None) -> Path:
    out = cfg.get("outputs") or {}
    base = Path(out.get("figures", Path(outputs_root or "outputs") / "figures"))
    return base / "maps"


def maps_traces_root(cfg: dict, outputs_root: Path | None = None) -> Path:
    out = cfg.get("outputs") or {}
    root = Path(out.get("root", outputs_root or "outputs"))
    return Path(out.get("maps_traces", root / "traces" / "maps"))


def _safe_name(instance_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", instance_name)


def snapshot_dir(figures_root: Path, instance_name: str, epoch: int) -> Path:
    return figures_root / _safe_name(instance_name) / f"epoch_{epoch}"


def list_training_instances(df: pd.DataFrame) -> list[str]:
    if df.empty or "instance_name" not in df.columns:
        return []
    return sorted(df["instance_name"].astype(str).unique().tolist())


def instance_epoch_rows(df: pd.DataFrame, instance_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[df["instance_name"].astype(str) == str(instance_name)].copy()
    if "epoch" in sub.columns:
        sub = sub.sort_values("epoch")
    return sub


def instance_progression_records(df: pd.DataFrame, instance_name: str) -> list[dict[str, Any]]:
    sub = instance_epoch_rows(df, instance_name)
    if sub.empty:
        return []
    obj_col = "final_objective" if "final_objective" in sub.columns else "cost"
    records = []
    for _, row in sub.iterrows():
        records.append({
            "epoch": int(row.get("epoch", 0)),
            "cycle": int(row.get("cycle", 0)) if "cycle" in row else None,
            "served_rate": float(row.get("served_rate", row.get("served_ratio", 0)) or 0),
            "objective": float(row.get(obj_col, 0) or 0),
            "feasibility_rate": float(row.get("feasibility_rate", 0) or 0),
            "unserved_customers": float(row.get("unserved_customers", 0) or 0),
        })
    return records


def log_row_for_epoch(df: pd.DataFrame, instance_name: str, epoch: int) -> dict[str, Any] | None:
    sub = instance_epoch_rows(df, instance_name)
    if sub.empty or "epoch" not in sub.columns:
        return None
    match = sub[sub["epoch"].astype(int) == int(epoch)]
    if match.empty:
        return None
    return match.iloc[-1].to_dict()


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


def _history_checkpoints(cfg: dict) -> list[tuple[int, Path]]:
    ckpt_dir = Path((cfg.get("outputs") or {}).get("checkpoints", "outputs/checkpoints"))
    hist = ckpt_dir / "history"
    out: list[tuple[int, Path]] = []
    if hist.is_dir():
        for path in hist.glob("epoch_*.pt"):
            try:
                ep = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if path.stat().st_size > 0:
                out.append((ep, path))
    out.sort(key=lambda x: x[0])
    return out


def resolve_checkpoint_for_epoch(cfg: dict, target_epoch: int) -> tuple[Path | None, int | None, str]:
    """Pick the newest history checkpoint at or before ``target_epoch``."""
    target_epoch = int(target_epoch)
    chosen_path: Path | None = None
    chosen_ep: int | None = None
    for ep, path in _history_checkpoints(cfg):
        if ep <= target_epoch:
            chosen_path = path
            chosen_ep = ep
        else:
            break
    if chosen_path is not None:
        note = f"history checkpoint epoch {chosen_ep}"
        if chosen_ep != target_epoch:
            note += f" (requested logged epoch {target_epoch})"
        return chosen_path, chosen_ep, note

    ckpt_dir = Path((cfg.get("outputs") or {}).get("checkpoints", "outputs/checkpoints"))
    for name in ("last.pt", "best.pt"):
        path = ckpt_dir / name
        if path.is_file() and path.stat().st_size > 0:
            meta = load_checkpoint_training_meta(path) or {}
            ckpt_ep = meta.get("epoch")
            note = f"{name}"
            if ckpt_ep is not None:
                note += f" (saved at epoch {ckpt_ep})"
            note += f" — no history checkpoint ≤ epoch {target_epoch}"
            return path, int(ckpt_ep) if ckpt_ep is not None else None, note
    return None, None, "no checkpoint available"


def _snapshot_meta_path(out_dir: Path) -> Path:
    return out_dir / "snapshot_meta.json"


def snapshot_is_cached(out_dir: Path) -> bool:
    if not out_dir.is_dir():
        return False
    meta = _snapshot_meta_path(out_dir)
    if not meta.is_file():
        return False
    for filename, _, _ in LIVE_PLOT_SPECS:
        png = out_dir / filename
        if not png.is_file() or png.stat().st_size == 0:
            return False
    return True


def list_snapshot_figures(out_dir: Path) -> list[dict[str, Any]]:
    items = []
    for filename, title, _ in LIVE_PLOT_SPECS:
        path = out_dir / filename
        exists = path.is_file() and path.stat().st_size > 0
        rel = f"{out_dir.name}/{filename}"
        # URL built by UI from instance/epoch
        items.append({
            "file": filename,
            "title": title,
            "exists": exists,
            "mtime": int(path.stat().st_mtime) if exists else 0,
        })
    return items


def figure_url(instance_name: str, epoch: int, filename: str, mtime: int) -> str:
    inst = _safe_name(instance_name)
    return f"{MAPS_MOUNT}/{inst}/epoch_{epoch}/{filename}?v={mtime}"


def generate_instance_snapshot(
    cfg: dict,
    df: pd.DataFrame,
    instance_name: str,
    epoch: int,
    *,
    figures_root: Path,
    traces_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run traced rollout + write diagnostic PNGs for one logged instance/epoch."""
    mcfg = maps_cfg(cfg)
    epoch = int(epoch)
    log_row = log_row_for_epoch(df, instance_name, epoch)
    if log_row is None:
        return {"ok": False, "error": f"No logged row for {instance_name} at epoch {epoch}."}

    out_dir = snapshot_dir(figures_root, instance_name, epoch)
    if not force and mcfg.get("use_cached_if_present", True) and snapshot_is_cached(out_dir):
        meta = json.loads(_snapshot_meta_path(out_dir).read_text(encoding="utf-8"))
        meta["ok"] = True
        meta["cached"] = True
        meta["figures"] = list_snapshot_figures(out_dir)
        meta["figures_dir"] = str(out_dir)
        return meta

    ckpt_path, ckpt_epoch, ckpt_note = resolve_checkpoint_for_epoch(cfg, epoch)
    if ckpt_path is None:
        return {"ok": False, "error": "No checkpoint found for this epoch."}

    train_ds = load_training_dataset(cfg)
    if instance_name not in train_ds.names:
        return {"ok": False, "error": f"Instance {instance_name} not in training dataset."}
    inst_idx = train_ds.names.index(instance_name)
    instance = train_ds[inst_idx]

    prob = cfg.get("problem") or {}
    decode = cfg.get("decoding") or {}
    hard_tw = str((cfg.get("constraints") or {}).get("time_windows_mode", "")) == "hard_original"
    device = "cpu"
    policy = load_policy(ckpt_path, device=device)

    import random

    fleet_rng = random.Random(int(cfg.get("seed", 42)) + inst_idx + epoch)
    m = max(1, int(round(float(log_row.get("avg_used_evs", 0))))) if log_row.get("avg_used_evs") else sample_fleet_size(instance, 3, fleet_rng)
    weights = CostWeights.from_config(cfg["cost"])

    _, trace = run_traced_episode(
        instance,
        policy,
        m,
        weights,
        mode=str(mcfg.get("trace_mode", decode.get("mode", "greedy"))),
        device=device,
        charging_portions=policy.charging_portions,
        use_horizon_check=bool(prob.get("use_horizon_check", True)),
        use_depot_charging=bool(prob.get("use_depot_charging", True)),
        max_steps_factor=int(decode.get("max_steps_factor", 6)),
        hard_time_windows=hard_tw,
        selection_tag=f"maps_{_safe_name(instance_name)}_e{epoch}",
        log_row={**log_row, "instance_name": instance_name, "instance_idx": inst_idx},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    generate_live_diagnostic_figures(trace, out_dir)
    traces_root.mkdir(parents=True, exist_ok=True)
    trace_path = traces_root / f"{_safe_name(instance_name)}_epoch_{epoch}.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    tmeta = trace.get("meta") or {}
    result = {
        "ok": True,
        "cached": False,
        "instance_name": instance_name,
        "epoch": epoch,
        "cycle": int(log_row.get("cycle", tmeta.get("cycle", 0))),
        "checkpoint": ckpt_path.name,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_note": ckpt_note,
        "objective": float(tmeta.get("objective", 0)),
        "served_ratio": float(tmeta.get("served_ratio", 0)),
        "feasible": bool(tmeta.get("feasible", False)),
        "unserved_customers": int(tmeta.get("unserved_customers", 0)),
        "used_evs": int(tmeta.get("used_evs", 0)),
        "total_distance": float(tmeta.get("total_distance", 0)),
        "trace_path": str(trace_path),
        "figures_dir": str(out_dir),
        "figures": list_snapshot_figures(out_dir),
    }
    _snapshot_meta_path(out_dir).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
