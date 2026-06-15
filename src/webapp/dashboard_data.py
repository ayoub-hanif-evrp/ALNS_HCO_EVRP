"""Load and prepare persisted training metrics for the dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

STALE_AFTER_SECONDS = 20.0


def read_log(log_path: Path) -> list[dict]:
    """Parse the JSONL training log; tolerate partially-written last lines."""
    if not log_path.exists():
        return []
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_metrics_df(log_path: Path) -> pd.DataFrame:
    rows = read_log(log_path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def prepare_metrics_df(df: pd.DataFrame, *, window: int = 10) -> pd.DataFrame:
    """Add rolling means for smoother KPI display."""
    if df.empty:
        return df
    out = df.copy()
    if "epoch" not in out.columns:
        out["epoch"] = range(len(out))
    roll_cols = [
        "reward",
        "cost",
        "feasibility_rate",
        "avg_used_evs",
        "avg_unserved",
        "avg_violation",
    ]
    w = max(1, min(window, len(out)))
    for col in roll_cols:
        if col in out.columns:
            out[f"rolling_{col}_mean"] = out[col].rolling(window=w, min_periods=1).mean()
    # Smoothed reward and best-so-far cost for stability views.
    if "reward" in out.columns:
        out["reward_smooth"] = out["reward"].ewm(span=max(2, w), adjust=False).mean()
    if "cost" in out.columns:
        out["cost_best"] = out["cost"].cummin()
    return out


def _latest_run_dir(outputs_root: Path) -> Path | None:
    runs = outputs_root / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(
        (p for p in runs.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_run_meta(outputs_root: Path | None = None) -> dict:
    outputs_root = outputs_root or Path("outputs")
    run_dir = _latest_run_dir(outputs_root)
    if run_dir is None:
        return {"run_id": "—", "run_dir": None}
    meta = {"run_id": run_dir.name, "run_dir": str(run_dir)}
    summary_path = run_dir / "final_summary.json"
    if summary_path.exists():
        try:
            meta.update(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return meta


def load_run_config(outputs_root: Path | None = None, fallback: Path | None = None) -> dict:
    outputs_root = outputs_root or Path("outputs")
    run_dir = _latest_run_dir(outputs_root)
    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    if fallback is not None and fallback.exists():
        from src.utils.config import load_config

        return load_config(fallback)
    return {}


def training_plan_from_config(cfg: dict) -> dict:
    tcfg = cfg.get("training") or {}
    return {
        "total_epochs": int(tcfg.get("epochs", 0) or 0),
        "batch_size": int(tcfg.get("batch_size", 0) or 0),
        "method": str(tcfg.get("method", "reinforce")),
        "learning_rate": float(tcfg.get("learning_rate", 0.0) or 0.0),
    }


def load_status(log_path: Path, cfg: dict) -> dict:
    plan = training_plan_from_config(cfg)
    total = int(plan["total_epochs"])
    rows = read_log(log_path)
    cur = int(rows[-1].get("epoch", len(rows) - 1)) if rows else 0
    age = time.time() - log_path.stat().st_mtime if log_path.exists() else float("inf")
    running = log_path.exists() and age < STALE_AFTER_SECONDS
    return {
        "epoch": cur,
        "total_epochs": total,
        "epochs_logged": len(rows),
        "running": running,
        "seconds_since_update": age if log_path.exists() else None,
        "message": (
            f"Training live · epoch {cur} · {len(rows)} logged"
            if running and rows
            else (
                f"Idle ({age:.0f}s since last update) · epoch {cur}"
                if rows
                else "Waiting for training to start…"
            )
        ),
    }


def progress_from_metrics(df: pd.DataFrame, plan: dict) -> dict:
    if df.empty:
        return {}
    cur = int(df["epoch"].max()) if "epoch" in df.columns else len(df) - 1
    total = int(plan.get("total_epochs") or 0)
    return {"epoch": cur, "progress": min(cur / max(total, 1), 1.0)}
