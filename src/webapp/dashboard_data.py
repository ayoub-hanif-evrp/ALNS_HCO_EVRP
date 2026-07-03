"""Load training status and metrics for the live dashboard (saved logs only)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

STALE_AFTER_SECONDS = 20.0
ROLLING_WINDOW = 20
DEFAULT_MA_WINDOW = 20

# Weighted objective / penalty columns logged during training (matches compute_cost breakdown).
COST_COLUMNS = [
    "mean_obj_vehicle",
    "mean_obj_distance",
    "mean_obj_duration",
    "mean_obj_charging",
    "mean_obj_charging_price",
    "mean_obj_waiting",
    "mean_pen_unserved",
    "mean_pen_late",
    "mean_pen_battery",
    "mean_pen_capacity",
    "mean_pen_depot",
    "mean_pen_reserved",
    "mean_pen_duplicate",
    "mean_pen_infeasible",
]

# One entry per term in J(π) — used for dedicated dashboard charts and KPI tiles.
COST_OBJECTIVE_TERMS: list[dict[str, str | None]] = [
    {
        "key": "vehicle",
        "col": "mean_obj_vehicle",
        "raw_col": "avg_used_evs",
        "label": "Fleet · λ_v × used EVs",
        "short": "Fleet",
        "formula": "λ_vehicle × number of used EVs",
        "ylabel": "Weighted term",
        "raw_ylabel": "Used EVs",
    },
    {
        "key": "distance",
        "col": "mean_obj_distance",
        "raw_col": "avg_distance",
        "label": "Distance · λ_d × total distance",
        "short": "Distance",
        "formula": "λ_distance × route distance",
        "ylabel": "Weighted term",
        "raw_ylabel": "Total distance",
    },
    {
        "key": "duration",
        "col": "mean_obj_duration",
        "raw_col": "avg_duration",
        "label": "Duration · λ_t × route time",
        "short": "Duration",
        "formula": "λ_duration × sum of EV completion times",
        "ylabel": "Weighted term",
        "raw_ylabel": "Total duration",
    },
    {
        "key": "charging",
        "col": "mean_obj_charging",
        "raw_col": "avg_charging_time",
        "label": "Charging time · λ_c × charge time",
        "short": "Charging time",
        "formula": "λ_charging × total charging time",
        "ylabel": "Weighted term",
        "raw_ylabel": "Charging time",
    },
    {
        "key": "charging_price",
        "col": "mean_obj_charging_price",
        "raw_col": "avg_charging_price",
        "label": "Charging price · λ_p × energy cost",
        "short": "Charging price",
        "formula": "λ_charging_price × station price × charged energy",
        "ylabel": "Weighted term",
        "raw_ylabel": "Charging price",
    },
    {
        "key": "waiting",
        "col": "mean_obj_waiting",
        "raw_col": "avg_waiting_time",
        "label": "Waiting · λ_w × waiting time",
        "short": "Waiting",
        "formula": "λ_waiting × customer waiting time",
        "ylabel": "Weighted term",
        "raw_ylabel": "Waiting time",
    },
]

COST_PENALTY_TERMS: list[dict[str, str | None]] = [
    {
        "key": "unserved",
        "col": "mean_pen_unserved",
        "raw_col": "avg_unserved",
        "label": "Unserved customers",
        "short": "Unserved",
        "formula": "penalty_unserved × count of unserved customers",
        "ylabel": "Penalty",
        "raw_ylabel": "Unserved count",
    },
    {
        "key": "late",
        "col": "mean_pen_late",
        "raw_col": None,
        "label": "Time-window lateness",
        "short": "Lateness",
        "formula": "penalty_late × total lateness",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "battery",
        "col": "mean_pen_battery",
        "raw_col": None,
        "label": "Battery violation",
        "short": "Battery",
        "formula": "penalty_battery × battery shortfall",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "capacity",
        "col": "mean_pen_capacity",
        "raw_col": None,
        "label": "Load capacity violation",
        "short": "Capacity",
        "formula": "penalty_capacity × capacity overload",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "depot",
        "col": "mean_pen_depot",
        "raw_col": None,
        "label": "Depot return violation",
        "short": "Depot",
        "formula": "penalty_depot × EVs not returning to depot",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "reserved",
        "col": "mean_pen_reserved",
        "raw_col": None,
        "label": "Reserved customer unfinished",
        "short": "Reserved",
        "formula": "penalty_reserved × unfinished reservations",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "duplicate",
        "col": "mean_pen_duplicate",
        "raw_col": None,
        "label": "Duplicate customer visit",
        "short": "Duplicate",
        "formula": "penalty_duplicate × duplicate visits",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
    {
        "key": "infeasible",
        "col": "mean_pen_infeasible",
        "raw_col": None,
        "label": "Global infeasibility flag",
        "short": "Infeasible",
        "formula": "penalty_infeasible × infeasible solution flag",
        "ylabel": "Penalty",
        "raw_ylabel": None,
    },
]


def read_log(log_path: Path) -> list[dict]:
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


def ma_window_from_config(cfg: dict) -> int:
    """Moving-average window = training instances per cycle (one full pass)."""
    ds = cfg.get("dataset") or {}
    train_n = len(ds.get("train_instances") or [])
    if train_n > 0:
        return train_n
    tcfg = cfg.get("training") or {}
    return int(tcfg.get("dashboard_ma_window", DEFAULT_MA_WINDOW))


def filter_training_instances(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only training instances; exclude held-out eval/test instances."""
    if df.empty:
        return df
    ds = cfg.get("dataset") or {}
    train_inst = list(ds.get("train_instances") or [])
    holdout = list(ds.get("eval_instances") or [])
    if train_inst and "instance_name" in df.columns:
        return df[df["instance_name"].isin(train_inst)].reset_index(drop=True)
    if holdout and "instance_name" in df.columns:
        return df[~df["instance_name"].isin(holdout)].reset_index(drop=True)
    return df


def cycle_column(df: pd.DataFrame) -> pd.Series:
    """Training cycle (pass) index — one full sweep over training instances."""
    if "pass" in df.columns and df["pass"].notna().any():
        return df["pass"].astype(int)
    if "cycle" in df.columns and df["cycle"].notna().any():
        return df["cycle"].astype(int)
    n_inst = int(df["instance_name"].nunique()) if "instance_name" in df.columns else 20
    n_inst = max(n_inst, 1)
    return pd.Series((df.index // n_inst) + 1, index=df.index)


def cycle_aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per training cycle."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["_cycle"] = cycle_column(work)
    agg_cols = {
        c: "mean"
        for c in [
            "cost",
            "final_objective",
            "feasibility_rate",
            "served_rate",
            "avg_violation",
            "unserved_customers",
            "avg_lateness",
            "avg_battery_violation",
            "avg_capacity_violation",
            "avg_depot_violation",
            "avg_duplicate_visits",
        ]
        if c in work.columns
    }
    if not agg_cols:
        return pd.DataFrame()
    return work.groupby("_cycle", as_index=False).agg(agg_cols).rename(columns={"_cycle": "cycle"})


def load_metrics_df(log_path: Path) -> pd.DataFrame:
    """Primary source: live JSONL; fall back to results CSV if JSONL is empty."""
    rows = read_log(log_path)
    if rows:
        return pd.DataFrame(rows)

    csv_path = Path(log_path).parent.parent / "results" / "training_metrics.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame()


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
    meta = {"run_id": run_dir.name, "run_dir": str(run_dir), "seed": "—"}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            meta["seed"] = cfg.get("seed", "—")
        except json.JSONDecodeError:
            pass
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
    ccfg = cfg.get("curriculum") or {}

    mode = str(ccfg.get("mode", "")).lower()
    if mode == "cyclic":
        cycles = int(ccfg.get("cycles", 5))
        class_order = [c.upper() for c in ccfg.get("class_order", ["C", "R", "RC"])]
        # Approximate total from config comment; exact count is in run summary / metrics length.
        per_cycle_hint = int(ccfg.get("steps_per_cycle_hint", 0)) or None
        total = (per_cycle_hint * cycles) if per_cycle_hint else 0
        return {
            "phases": [],
            "phase_names": {1: "C", 2: "R", 3: "RC"},
            "curriculum_stages": [1, 2, 3],
            "total_training_updates": total,
            "total_epochs": total,
            "batch_size": 1,
            "method": str(tcfg.get("method", "actor_critic")),
            "learning_rate": float(tcfg.get("learning_rate", 0.0) or 0.0),
            "training_mode": "cyclic",
            "cycles": cycles,
            "class_order": class_order,
        }

    curriculum = cfg.get("curriculum")
    if curriculum:
        from src.training.curriculum import phases_from_config

        phases = phases_from_config(curriculum)
        phase_rows = [
            {"id": p.index, "name": p.name, "max_episodes": p.epochs, "classes": p.classes}
            for p in phases
        ]
        total = sum(int(p.epochs) for p in phases)
        return {
            "phases": phase_rows,
            "phase_names": {int(p.index): p.name for p in phases},
            "curriculum_stages": [int(p.index) for p in phases],
            "total_training_updates": total,
            "total_epochs": total,
            "batch_size": int(tcfg.get("batch_size", 0) or 0),
            "method": str(tcfg.get("method", "actor_critic")),
            "learning_rate": float(tcfg.get("learning_rate", 0.0) or 0.0),
            "training_mode": "curriculum",
        }

    epochs = int(tcfg.get("epochs", 0) or 0)
    passes = int(tcfg.get("cycles") or tcfg.get("passes") or 0)
    train_names = list((cfg.get("dataset") or {}).get("train_instances") or [])
    n_train = len(train_names)
    if not epochs and passes and n_train:
        epochs = passes * n_train
    return {
        "phases": [],
        "phase_names": {1: "flat_actor_critic"},
        "curriculum_stages": [1],
        "total_training_updates": epochs,
        "total_epochs": epochs,
        "batch_size": int(tcfg.get("batch_size", 0) or 0),
        "method": str(tcfg.get("method", "actor_critic")),
        "learning_rate": float(tcfg.get("learning_rate", 0.0) or 0.0),
        "training_mode": "flat",
        "passes": passes,
        "cycles": passes,
        "n_train_instances": n_train,
        "train_instances": train_names,
        "ma_window": ma_window_from_config(cfg),
    }


def prepare_metrics_df(
    df: pd.DataFrame,
    *,
    window: int | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    if window is None:
        window = ma_window_from_config(cfg or {}) if cfg else ROLLING_WINDOW
    if df.empty:
        return df

    d = df.copy()
    if "epoch" in d.columns:
        d = d.sort_values("epoch", na_position="last")
        d = d.drop_duplicates(subset=["epoch"], keep="last")
    d = d.reset_index(drop=True)

    idx = d.index
    extras: dict[str, pd.Series] = {}
    extras["episode_index"] = pd.Series(range(1, len(d) + 1), index=idx)
    if "epoch" in d.columns:
        extras["training_update"] = d["epoch"]

    w = max(1, min(window, len(d)))

    def _resolve_col(col: str) -> pd.Series | None:
        if col in d.columns:
            return d[col]
        if col in extras:
            return extras[col]
        return None

    def _roll_mean(col: str, out_name: str) -> None:
        series = _resolve_col(col)
        if series is not None:
            extras[out_name] = series.rolling(window=w, min_periods=1).mean()

    def _roll_std(col: str, out_name: str) -> None:
        series = _resolve_col(col)
        if series is not None:
            extras[out_name] = series.rolling(window=w, min_periods=1).std().fillna(0.0)

    # Reference-compatible aliases (batched — avoids fragmented frame warnings).
    if "served_rate" in d.columns:
        extras["served_ratio"] = d["served_rate"]
    if "feasibility_rate" in d.columns:
        extras["mission_complete_rate"] = d["feasibility_rate"]
        extras["infeasibility_rate"] = 1.0 - d["feasibility_rate"]
    if "reward" in d.columns:
        extras["raw_episode_return"] = d["reward"]
        extras["ppo_effective_return"] = d["reward"]
    if "advantage" in d.columns:
        extras["phase_relative_return"] = d["advantage"]
    if "mean_pen_late" in d.columns:
        extras["cost_tardiness"] = d["mean_pen_late"]
        extras["total_tardiness"] = d["mean_pen_late"]
    if "avg_waiting_time" in d.columns:
        extras["total_early_waiting"] = d["avg_waiting_time"]
    if "avg_charging_time" in d.columns:
        extras["avg_charging_duration"] = d["avg_charging_time"]
        extras["charging_visits"] = (d["avg_charging_time"] > 1e-6).astype(float)
    if "distance_per_ev" in d.columns:
        extras["avg_distance_per_ev"] = d["distance_per_ev"]
    if "mean_pen_capacity" in d.columns:
        extras["capacity_violations"] = d["mean_pen_capacity"]
    if "mean_pen_battery" in d.columns:
        extras["battery_violations"] = d["mean_pen_battery"]
    if "mean_pen_duplicate" in d.columns:
        extras["duplicate_prevented"] = d["mean_pen_duplicate"]
    if "mean_pen_depot" in d.columns:
        extras["depot_returns"] = d["mean_pen_depot"]
    if "avg_unserved" in d.columns:
        extras["cap_hit_rate"] = (d["avg_unserved"] > 0.5).astype(float)
    if "phase" in d.columns and "training_phase_id" not in d.columns:
        extras["training_phase_id"] = d["phase"]
    if "phase_name" in d.columns:
        extras["training_phase_name"] = d["phase_name"]

    extras["training_cycle"] = cycle_column(d)

    if "mean_feasible_actions" in d.columns and "num_nodes" in d.columns and "feasible_action_ratio" not in d.columns:
        extras["feasible_action_ratio"] = d["mean_feasible_actions"] / d["num_nodes"].clip(lower=1)

    # Canonical class column for consistent chart colors (C / R / RC).
    if "instance_class" in d.columns:
        instance_class = d["instance_class"].astype(str).str.upper()
    elif "training_phase_id" in extras:
        inv = {v: k for k, v in {"C": 1, "R": 2, "RC": 3}.items()}
        instance_class = extras["training_phase_id"].map(lambda x: inv.get(int(x), "?"))
    elif "training_phase_id" in d.columns:
        inv = {v: k for k, v in {"C": 1, "R": 2, "RC": 3}.items()}
        instance_class = d["training_phase_id"].map(lambda x: inv.get(int(x), "?"))
    elif "phase" in d.columns:
        inv = {v: k for k, v in {"C": 1, "R": 2, "RC": 3}.items()}
        instance_class = d["phase"].map(lambda x: inv.get(int(x), "?"))
    else:
        instance_class = pd.Series("?", index=idx, dtype=object)
    extras["instance_class"] = instance_class

    from src.webapp.plot_style import CLASS_STAGE_ID

    extras["class_stage_id"] = instance_class.map(lambda c: CLASS_STAGE_ID.get(str(c).upper(), 0))

    w_cls = max(3, min(window, 15))

    def _roll_by_class(col: str, stat: str = "mean") -> None:
        series = _resolve_col(col)
        if series is None:
            return
        for cls in instance_class.unique():
            cls = str(cls).upper()
            if cls not in ("C", "R", "RC"):
                continue
            mask = instance_class == cls
            roll = series.loc[mask].rolling(window=min(w_cls, int(mask.sum())), min_periods=1)
            out = pd.Series(float("nan"), index=idx, dtype=float)
            if stat == "mean":
                out.loc[mask] = roll.mean().to_numpy()
            else:
                out.loc[mask] = roll.std().fillna(0.0).to_numpy()
            extras[f"rolling_{col}_{cls}"] = out

    class_roll_cols = [
        "reward",
        "raw_episode_return",
        "cost",
        "served_ratio",
        "avg_used_evs",
        "loss",
        "policy_loss",
        "advantage",
        "entropy",
        "grad_norm",
        "learning_rate",
        "feasibility_rate",
        "avg_unserved",
        "avg_distance",
        "distance_per_customer",
        "avg_charging_time",
        "charging_time_share",
        "mean_pen_battery",
        "mean_pen_late",
        "mean_obj_waiting",
        "mean_pen_capacity",
        "mean_pen_reserved",
        "capacity_violations",
        "battery_violations",
    ]
    for col in class_roll_cols:
        _roll_by_class(col)

    # Per-class latest snapshot (for bar charts).
    served = extras.get("served_ratio")
    for cls in ("C", "R", "RC"):
        mask = instance_class == cls
        if not mask.any():
            continue
        tail = d.loc[mask].tail(min(w_cls, int(mask.sum())))
        extras[f"latest_mean_reward_{cls}"] = pd.Series(
            float(tail["reward"].mean()) if "reward" in tail else float("nan"), index=idx
        )
        extras[f"latest_mean_cost_{cls}"] = pd.Series(
            float(tail["cost"].mean()) if "cost" in tail else float("nan"), index=idx
        )
        extras[f"latest_mean_served_{cls}"] = pd.Series(
            float(served.loc[tail.index].mean()) if served is not None else float("nan"),
            index=idx,
        )
        extras[f"latest_mean_evs_{cls}"] = pd.Series(
            float(tail["avg_used_evs"].mean()) if "avg_used_evs" in tail else float("nan"), index=idx
        )
        extras[f"latest_mean_feas_{cls}"] = pd.Series(
            float(tail["feasibility_rate"].mean()) if "feasibility_rate" in tail else float("nan"), index=idx
        )

    _roll_mean("reward", "rolling_raw_return_mean")
    _roll_std("reward", "rolling_raw_return_std")
    _roll_mean("advantage", "rolling_phase_relative_return_mean")
    _roll_std("advantage", "rolling_phase_relative_return_std")
    _roll_mean("served_ratio", "rolling_served_ratio_mean")
    _roll_mean("mission_complete_rate", "rolling_mission_complete_rate")
    _roll_mean("infeasibility_rate", "rolling_infeasible_rate")
    _roll_mean("cap_hit_rate", "rolling_cap_rate")
    _roll_mean("feasibility_rate", "rolling_feasibility_rate_mean")
    _roll_mean("cost", "rolling_cost_mean")
    _roll_mean("avg_used_evs", "rolling_avg_used_evs_mean")
    _roll_mean("avg_unserved", "rolling_avg_unserved_mean")
    _roll_mean("avg_violation", "rolling_avg_violation_mean")

    global_roll_cols = [
        "final_objective",
        "raw_objective_return",
        "shaped_training_return",
        "served_bonus_total",
        "early_depot_penalty_total",
        "step_penalty_total",
        "terminal_unserved_penalty_scaled",
        "terminal_penalty_coef",
        "relative_objective_mismatch",
        "fraction_customer_actions",
        "fraction_station_actions",
        "fraction_depot_actions",
        "early_depot_returns",
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_param_delta_norm",
        "critic_param_delta_norm",
        "advantage_std",
        "loss",
        "policy_loss",
        "actor_loss",
        "critic_loss",
        "total_loss",
        "q_critic_loss",
        "advantage",
        "entropy",
        "grad_norm",
        "learning_rate",
        "critic_explained_variance",
        "avg_distance",
        "avg_duration",
        "avg_charging_price",
        "distance_per_customer",
        "avg_charging_time",
        "charging_time_share",
        "avg_waiting_time",
        "route_balance_cv",
        "charging_station_visits",
        "min_remaining_battery",
        "feasible_action_ratio",
        "unserved_customers",
        "avg_lateness",
        "avg_battery_violation",
        "avg_capacity_violation",
        "avg_depot_violation",
        "avg_duplicate_visits",
        *COST_COLUMNS,
    ]
    for col in global_roll_cols:
        _roll_mean(col, f"rolling_{col}_mean")

    if "reward" in d.columns and "best_reward" not in d.columns:
        extras["best_reward"] = d["reward"].cummax()
    if "cost" in d.columns:
        extras["cost_best"] = d["cost"].cummin()
    obj_col = "final_objective" if "final_objective" in d.columns else ("cost" if "cost" in d.columns else None)
    if obj_col and "best_objective_so_far" not in d.columns:
        extras["best_objective_so_far"] = d[obj_col].astype(float).cummin()

    # Overwrite existing columns in-place; concat only net-new columns (avoids duplicate labels).
    overlap = {k: extras.pop(k) for k in list(extras.keys()) if k in d.columns}
    if overlap:
        d = d.assign(**overlap)
    if extras:
        d = pd.concat([d, pd.DataFrame(extras, index=idx)], axis=1)
    return d


def _resolve_total_epochs(plan: dict, rows: list[dict], df_len: int = 0) -> int:
    """Use config estimate, but expand to logged count when training finished."""
    total = int(plan.get("total_epochs") or plan.get("total_training_updates") or 0)
    logged = max(df_len, len(rows))
    if rows:
        cur = int(rows[-1].get("epoch", logged - 1))
        logged = max(logged, cur + 1)
    if logged > total:
        return logged
    return total if total > 0 else logged


def load_status(log_path: Path, cfg: dict) -> dict:
    plan = training_plan_from_config(cfg)
    rows = read_log(log_path)
    total = _resolve_total_epochs(plan, rows)
    cur = int(rows[-1].get("epoch", len(rows) - 1)) if rows else 0
    age = time.time() - log_path.stat().st_mtime if log_path.exists() else float("inf")
    running = log_path.exists() and age < STALE_AFTER_SECONDS
    phase_name = str(rows[-1].get("curriculum_stage") or rows[-1].get("phase_name", "")) if rows else ""
    phase_id = int(rows[-1].get("phase", 0)) if rows else 0
    finished = bool(rows) and not running and (cur + 1 >= total or len(rows) >= total)
    return {
        "epoch": cur,
        "training_update": cur,
        "total_epochs": total,
        "total_training_updates": total,
        "epochs_logged": len(rows),
        "running": running,
        "finished": finished,
        "curriculum_stage": phase_id,
        "training_phase_name": phase_name,
        "stage_name": phase_name,
        "seconds_since_update": age if log_path.exists() else None,
        "phase": "training" if running else ("finished" if finished else "waiting"),
        "message": (
            f"Training live · epoch {cur} / {total}"
            if running and rows
            else (
                f"Training complete · {len(rows)} epochs logged"
                if finished
                else (
                    f"Idle ({age:.0f}s since last update) · epoch {cur} / {total}"
                    if rows
                    else "Waiting for training to start…"
                )
            )
        ),
    }


def progress_from_metrics(df: pd.DataFrame, plan: dict) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    total = int(plan.get("total_training_updates") or plan.get("total_epochs") or len(df))
    cur = int(last.get("training_update", last.get("epoch", len(df) - 1)))
    logged = len(df)
    if logged > total:
        total = logged
    if cur + 1 > total:
        total = cur + 1
    phase_name = str(
        last.get("curriculum_stage") or last.get("training_phase_name") or last.get("phase_name") or ""
    )
    if plan.get("training_mode") == "cyclic":
        phase_name = str(last.get("instance_class") or phase_name)
        cycle = int(last.get("cycle") or 0)
        if cycle:
            phase_name = f"cycle {cycle} · {phase_name}"
    elif plan.get("training_mode") == "flat":
        cycle = int(last.get("pass") or last.get("training_cycle") or 0)
        inst = str(last.get("instance_name") or "")
        if cycle and inst:
            phase_name = f"pass {cycle} · {inst}"
        elif inst:
            phase_name = inst
    return {
        "epoch": cur,
        "training_update": cur,
        "total_epochs": total,
        "total_training_updates": total,
        "overall_progress": min(cur / max(total, 1), 1.0),
        "curriculum_stage": int(last.get("training_phase_id") or last.get("phase") or 0),
        "training_phase_name": phase_name,
        "stage_name": phase_name,
    }
