"""E-VRPTW Transformer — training progress monitor (NiceGUI).

Reads only persisted logs under outputs/logs/train_log.jsonl and archived runs
under outputs/runs/.
"""

from __future__ import annotations

# Must run before importing nicegui (restores pkgutil.find_loader on 3.14).
from . import _compat  # noqa: F401

from pathlib import Path

from nicegui import ui

from src.webapp.dashboard_charts import TRAINING_CHART_SECTIONS, build_chart_figure
from src.webapp.dashboard_data import (
    load_metrics_df,
    load_run_config,
    load_run_meta,
    load_status,
    prepare_metrics_df,
    progress_from_metrics,
    training_plan_from_config,
)
from src.webapp.plot_style import SECTION_ACCENTS

_PAGE = "w-full max-w-[1280px] mx-auto px-6"
_SECTION = "w-full rounded-xl border border-slate-200 bg-white shadow-sm mb-5"
_SECTION_HEAD = "px-5 py-3 border-b border-slate-100 bg-slate-50"
_SECTION_BODY = "p-4"
_TILE = "rounded-lg border border-slate-100 bg-white p-3"


def _fmt_pct(val) -> str:
    try:
        return f"{100.0 * float(val):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(val, digits: int = 3) -> str:
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _field(label: str, value: str, *, mono: bool = False) -> None:
    ui.label(label).classes("text-[11px] uppercase tracking-wide text-slate-400")
    ui.label(value).classes(
        "text-sm font-semibold text-slate-800" + (" font-mono" if mono else "")
    )


def _kpi(label: str, value: str) -> None:
    with ui.column().classes(
        "flex-1 min-w-[8.5rem] items-center py-3 border-r border-slate-100 last:border-0"
    ):
        ui.label(label).classes("text-[11px] uppercase tracking-wide text-slate-400")
        ui.label(value).classes("text-lg font-semibold text-slate-900")


def _section_has_data(df, charts: list) -> bool:
    if df.empty:
        return False
    cols = set(df.columns)
    for spec in charts:
        ctype = spec.get("type")
        if ctype == "area":
            series = spec.get("series") or []
            if any(s[0] in cols for s in series):
                return True
        if ctype == "multi_line":
            series = spec.get("series") or []
            if any(s[0] in cols for s in series):
                return True
        if spec.get("col") in cols:
            return True
    return False


def _plotly_payload(fig) -> dict:
    """Plotly dict with responsive config for NiceGUI."""
    payload = fig.to_plotly_json()
    layout = payload.setdefault("layout", {})
    layout["autosize"] = True
    layout.pop("width", None)
    cfg = payload.setdefault("config", {})
    cfg["responsive"] = True
    cfg["displayModeBar"] = False
    return payload


def _plot_chart(df, plan: dict, spec: dict) -> None:
    fig = build_chart_figure(df, plan, spec)
    if fig is None:
        return
    title = spec.get("title", spec.get("col", "Chart"))
    desc = spec.get("desc")
    with ui.element("div").classes("chart-slot").style("width: 100%; min-width: 0;"):
        with ui.column().classes(_TILE).style("width: 100%; box-sizing: border-box;"):
            ui.label(title).classes("text-sm font-medium text-slate-700 mb-1")
            if desc:
                ui.label(desc).classes("text-xs text-slate-400 mb-2")
            ui.plotly(_plotly_payload(fig)).classes("w-full").style(
                "width: 100%; display: block; min-width: 0;"
            )


def _render_charts(df, plan: dict, charts: list) -> None:
    """One full-width chart per row (reliable with NiceGUI + Plotly)."""
    with ui.column().classes("w-full gap-4").style("width: 100%;"):
        for spec in charts:
            _plot_chart(df, plan, spec)


def create_dashboard(
    log_path: str | Path,
    *,
    config_path: str | Path | None = None,
    outputs_root: str | Path | None = None,
) -> None:
    """Register the dashboard page bound to a training log path."""
    log_path = Path(log_path)
    config_path = Path(config_path) if config_path is not None else None
    outputs_root = Path(outputs_root) if outputs_root is not None else Path("outputs")

    @ui.page("/")
    def page() -> None:  # noqa: ANN202
        ui.colors(primary="#2563eb", positive="#16a34a", negative="#dc2626")
        ui.add_head_html(
            """
            <style>
              body { background: #f1f5f9 !important; }
              .nicegui-content { padding: 0 !important; }
              .chart-slot { width: 100% !important; min-width: 0 !important; box-sizing: border-box; }
              .chart-slot .js-plotly-plot,
              .chart-slot .plotly-graph-div,
              .chart-slot .plot-container {
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
              }
              .js-plotly-plot .modebar { display: none !important; }
            </style>
            """
        )

        status_panel = ui.column().classes(_PAGE + " pt-4")
        kpi_panel = ui.column().classes(_PAGE)
        charts_panel = ui.column().classes(_PAGE + " pb-12")

        def refresh() -> None:
            status = load_status(log_path, load_run_config(outputs_root, config_path))
            meta = load_run_meta(outputs_root)
            cfg = load_run_config(outputs_root, config_path)
            df = prepare_metrics_df(load_metrics_df(log_path))
            plan = training_plan_from_config(cfg)
            if not df.empty:
                status = {**status, **progress_from_metrics(df, plan)}

            total = int(plan.get("total_epochs") or status.get("total_epochs") or 0)
            cur = int(status.get("epoch") or 0)
            running = bool(status.get("running"))

            status_panel.clear()
            with status_panel:
                with ui.card().classes(_SECTION):
                    with ui.element("div").classes(
                        "grid grid-cols-1 lg:grid-cols-3 lg:divide-x divide-slate-100"
                    ):
                        with ui.column().classes("p-5 gap-2"):
                            ui.label("Run").classes("text-xs font-bold text-slate-400")
                            _field("Run ID", str(meta.get("run_id", "—")), mono=True)
                            _field("Seed", str(cfg.get("seed", "—")))
                            _field("Epochs logged", str(len(df)))
                            _field("Log file", str(log_path), mono=True)

                        with ui.column().classes("p-5 gap-2 lg:border-x border-slate-100"):
                            ui.label("Progress").classes("text-xs font-bold text-slate-400")
                            _field("Epoch", f"{cur} / {total or '—'}")
                            ui.linear_progress(value=min(cur / max(total, 1), 1.0)).classes("w-full")
                            if running:
                                ui.label("Training live").classes(
                                    "text-sm font-medium text-green-700"
                                )
                            elif not df.empty:
                                ui.label("Idle").classes("text-sm font-medium text-amber-700")
                            else:
                                ui.label("Waiting for data").classes(
                                    "text-sm font-medium text-slate-500"
                                )
                            _field("Method", str(plan.get("method", "reinforce")).upper())
                            _field("Batch size", str(plan.get("batch_size", "—")))
                            if status.get("message"):
                                ui.label(status["message"]).classes("text-xs text-slate-400")

                        with ui.column().classes("p-5 gap-2"):
                            ui.label("Training setup").classes("text-xs font-bold text-slate-400")
                            _field("Policy", "Multi-agent Transformer")
                            _field("Algorithm", "REINFORCE + moving baseline")
                            fleet = cfg.get("fleet_search") or {}
                            _field(
                                "Fleet search",
                                "enabled" if fleet.get("enabled") else "disabled",
                            )
                            ui.label(
                                "Results archived under outputs/runs/ and "
                                "outputs/results/archive/."
                            ).classes("text-xs text-slate-500")
                            ui.label("E-VRPTW · hard masks · charging head").classes(
                                "text-[10px] px-2 py-1 rounded-full"
                            ).style("background:#2563eb;color:#fff;")

            kpi_panel.clear()
            if not df.empty:
                last = df.iloc[-1]

                def pick(roll: str, raw: str):
                    if roll in df.columns:
                        v = last.get(roll)
                        if v is not None and v == v:
                            return v
                    return last.get(raw)

                with kpi_panel:
                    with ui.card().classes(_SECTION):
                        ui.label(
                            "Recent performance · last logged epoch "
                            "(rolling mean where available)"
                        ).classes("text-xs font-bold text-slate-400 px-5 pt-4")
                        with ui.row().classes("w-full flex-wrap justify-around"):
                            _kpi("Reward", _fmt_num(last.get("reward"), 1))
                            _kpi("Best reward", _fmt_num(last.get("best_reward"), 1))
                            _kpi("Cost J", _fmt_num(last.get("cost"), 1))
                            _kpi("Feasibility", _fmt_pct(pick("rolling_feasibility_rate_mean", "feasibility_rate")))
                            _kpi("Used EVs", _fmt_num(pick("rolling_avg_used_evs_mean", "avg_used_evs"), 2))
                            _kpi("Unserved", _fmt_num(pick("rolling_avg_unserved_mean", "avg_unserved"), 2))
                            _kpi("Violation", _fmt_num(pick("rolling_avg_violation_mean", "avg_violation"), 2))
                            _kpi("Distance", _fmt_num(last.get("avg_distance"), 1))
                            _kpi("Loss", _fmt_num(last.get("loss"), 3))
                            _kpi("Entropy", _fmt_num(last.get("entropy"), 3))

            charts_panel.clear()
            with charts_panel:
                if df.empty:
                    with ui.card().classes(_SECTION + " p-10 text-center"):
                        ui.label("Waiting for metrics…").classes("text-slate-600")
                        ui.label(
                            "python main_train.py --config configs/default.yaml"
                        ).classes("text-xs font-mono text-slate-400 mt-2")
                        ui.label(
                            "python main_dashboard.py --config configs/default.yaml"
                        ).classes("text-xs font-mono text-slate-400 mt-1")
                        ui.label("See commands.txt for the full command list").classes(
                            "text-xs text-slate-400 mt-1"
                        )
                    return

                for n, section in enumerate(TRAINING_CHART_SECTIONS, 1):
                    charts = section.get("charts", [])
                    if not _section_has_data(df, charts):
                        continue
                    accent = SECTION_ACCENTS[(n - 1) % len(SECTION_ACCENTS)]
                    with ui.card().classes(_SECTION):
                        with ui.row().classes(_SECTION_HEAD + " items-center gap-3"):
                            ui.element("div").classes("w-1.5 h-9 rounded-full shrink-0").style(
                                f"background:{accent}"
                            )
                            with ui.column().classes("gap-0"):
                                ui.label(section["title"]).classes(
                                    "text-base font-semibold text-slate-800"
                                )
                                if section.get("subtitle"):
                                    ui.label(section["subtitle"]).classes(
                                        "text-xs text-slate-500"
                                    )
                        with ui.column().classes(_SECTION_BODY).style("width: 100%;"):
                            _render_charts(df, plan, charts)

        with ui.header().classes("bg-white border-b border-slate-200 text-slate-800"):
            with ui.row().classes(
                "w-full max-w-[1280px] mx-auto px-6 py-3 items-center justify-between"
            ):
                with ui.column().classes("gap-0"):
                    ui.label("E-VRPTW Transformer · Training monitor").classes(
                        "text-lg font-bold"
                    )
                    ui.label(str(log_path)).classes("text-xs text-slate-400")
                with ui.row().classes("gap-3 items-center"):
                    ui.button("Refresh", on_click=refresh, icon="refresh").props("outline dense")
                    auto = ui.switch("Auto-refresh (3s)", value=True).props("dense")
            ui.label(
                "One chart per row · no mode bar · objective J = fleet + route costs + penalties · "
                "feasibility rate and unserved customers are the primary constraint indicators"
            ).classes(
                "text-[11px] text-slate-400 px-6 pb-2 w-full max-w-[1280px] mx-auto block"
            )

        poll = ui.timer(3.0, refresh)
        poll.active = True
        auto.on("update:model-value", lambda e: setattr(poll, "active", bool(e.value)))
        ui.timer(0.1, refresh, once=True)
