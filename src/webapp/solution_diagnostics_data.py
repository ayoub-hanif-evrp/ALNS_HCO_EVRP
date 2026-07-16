"""Solution-behavior diagnostic payloads for the live dashboard (no NiceGUI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.training.solution_diagnostics import (
    SOLUTION_DIAGNOSTIC_GROUPS,
    build_final_cycle_table_html,
)
from src.webapp.live_solution_diagnostics import ensure_live_random_diagnostics

FIGURES_MOUNT = "/training-figures"
TRACES_MOUNT = "/training-traces"


def figures_dir_from_config(cfg: dict, outputs_root: Path | None = None) -> Path:
    out = cfg.get("outputs") or {}
    root = Path(out.get("figures", Path(outputs_root or "outputs") / "figures"))
    return root / "training"


def traces_dir_from_config(cfg: dict, outputs_root: Path | None = None) -> Path:
    out = cfg.get("outputs") or {}
    if out.get("traces"):
        return Path(out["traces"])
    return Path(out.get("root", outputs_root or "outputs")) / "traces"


def _enrich_artifact(figures_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = figures_dir / item["file"]
    exists = path.is_file() and path.stat().st_size > 0
    return {
        **item,
        "exists": exists,
        "mtime": int(path.stat().st_mtime) if exists else 0,
    }


def build_solution_behavior_section(
    figures_dir: Path,
    df: pd.DataFrame,
    *,
    traces_dir: Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Any] | None:
    """Build live-dashboard section payload for solution behavior diagnostics."""
    cfg = cfg or {}
    dcfg = cfg.get("dashboard") or {}
    live_enabled = dcfg.get("live_random_diagnostics", True)
    static_enabled = dcfg.get("save_representative_traces", True)
    if not live_enabled and not static_enabled:
        return None

    figures_dir = Path(figures_dir)
    traces_dir = traces_dir or traces_dir_from_config(cfg)

    live = ensure_live_random_diagnostics(cfg, df, figures_dir, traces_dir) if live_enabled else {"enabled": False}

    static_groups = []
    any_static = False
    if static_enabled:
        for group in SOLUTION_DIAGNOSTIC_GROUPS:
            items = [_enrich_artifact(figures_dir, item) for item in group["items"]]
            any_static = any_static or any(i["exists"] for i in items)
            static_groups.append({**group, "items": items})

    table_html = build_final_cycle_table_html(df) if not df.empty else ""
    has_table = "<table" in table_html

    trace_files: list[dict[str, str]] = []
    if traces_dir.is_dir():
        for path in sorted(traces_dir.glob("trace_*.json")):
            trace_files.append({"name": path.name, "url": f"{TRACES_MOUNT}/{path.name}"})

    live_any = bool(live.get("any_figure"))
    if df.empty and not live_any and not any_static and not has_table:
        return None

    static_index = figures_dir / "index.html"
    live_state = live.get("state") or {}
    subtitle_parts = []
    if live_enabled:
        subtitle_parts.append(
            f"Live random training instance every ~{live.get('interval_sec', 45)}s "
            "(greedy rollout from latest checkpoint)."
        )
    if static_enabled and any_static:
        subtitle_parts.append("Post-training representative episodes below when available.")

    return {
        "id": "solution_behavior",
        "title": "11. Solution behavior diagnostics",
        "nav": "Solution behavior",
        "subtitle": " ".join(subtitle_parts) or "Route-level rollout diagnostics.",
        "type": "solution_diagnostics",
        "live": live,
        "groups": static_groups,
        "table_html": table_html if has_table else None,
        "trace_files": trace_files,
        "any_figure": live_any or any_static,
        "static_index_url": f"{FIGURES_MOUNT}/index.html" if static_index.is_file() else None,
        "figures_pending": live_enabled and not live_any and not any_static and not live.get("generating"),
    }
