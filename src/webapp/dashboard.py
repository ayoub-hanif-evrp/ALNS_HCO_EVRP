"""E-VRPTW Transformer — training progress monitor (NiceGUI)."""

from __future__ import annotations

from . import _compat  # noqa: F401

from pathlib import Path

from nicegui import ui

from src.webapp.dashboard_charts import TRAINING_CHART_SECTIONS, build_chart_figure, is_wide_chart
from src.webapp.dashboard_data import (
    load_metrics_df,
    load_run_config,
    load_run_meta,
    load_status,
    prepare_metrics_df,
    progress_from_metrics,
    training_plan_from_config,
)
from src.webapp.plot_style import CLASS_COLORS, CLASS_ORDER, STAGE_COLORS

_PAGE = "w-full max-w-[1280px] mx-auto px-6"
_SECTION = "w-full rounded-xl border border-slate-200 bg-white shadow-sm mb-5"
_SECTION_HEAD = "px-5 py-3 border-b border-slate-100 bg-slate-50"
_SECTION_BODY = "p-4"
_TILE = "rounded-lg border border-slate-100 bg-white p-3"
_ACCENTS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5"]


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
        "flex-1 min-w-[7rem] items-center py-2 px-1 border-r border-slate-100 last:border-0"
    ):
        ui.label(label).classes("text-[10px] uppercase tracking-wide text-slate-400 text-center")
        ui.label(value).classes("text-base font-semibold text-slate-900")


def _section_has_data(df, charts: list) -> bool:
    for c in charts:
        t = c.get("type", "metric")
        if t in ("cost_breakdown", "reward_components", "served_coverage", "optimization_panel"):
            return True
        if t == "per_class_bars":
            metrics = c.get("metrics") or []
            if any(m[0] in df.columns for m in metrics):
                return True
            continue
        if t == "rolling_band" and c.get("col") in df.columns:
            return True
        if t in ("class_bar", "cycle_bars", "class_metric") and c.get("col") in df.columns:
            return True
        if c.get("col") in df.columns:
            return True
    return False


def _plotly_payload(fig) -> dict:
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
    wide = [s for s in charts if is_wide_chart(s)]
    compact = [s for s in charts if s not in wide]
    with ui.column().classes("w-full gap-4").style("width: 100%;"):
        for spec in wide:
            _plot_chart(df, plan, spec)
    if compact:
        with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
            for spec in compact:
                _plot_chart(df, plan, spec)


def create_dashboard(
    log_path: str | Path,
    *,
    config_path: str | Path | None = None,
    outputs_root: str | Path | None = None,
) -> None:
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
              .chart-slot { width: 100% !important; min-width: 0 !important; }
              .chart-slot .js-plotly-plot, .chart-slot .plotly-graph-div { width: 100% !important; }
              .js-plotly-plot .modebar { display: none !important; }
            </style>
            """
        )

        status_panel = ui.column().classes(_PAGE + " pt-4")
        reward_kpi_panel = ui.column().classes(_PAGE)
        charts_panel = ui.column().classes(_PAGE + " pb-12")

        def refresh() -> None:
            cfg = load_run_config(outputs_root, config_path)
            status = load_status(log_path, cfg)
            meta = load_run_meta(outputs_root)
            df = prepare_metrics_df(load_metrics_df(log_path))
            plan = training_plan_from_config(cfg)

            if not df.empty and not plan.get("total_training_updates"):
                plan["total_training_updates"] = len(df)
                plan["total_epochs"] = len(df)

            if not df.empty:
                status = {**status, **progress_from_metrics(df, plan)}

            total = int(plan.get("total_training_updates") or status.get("total_epochs") or len(df) or 0)
            cur = int(status.get("training_update") or status.get("epoch") or 0)
            is_cyclic = plan.get("training_mode") == "cyclic"

            # ---- Status ----
            status_panel.clear()
            with status_panel:
                with ui.card().classes(_SECTION):
                    with ui.element("div").classes(
                        "grid grid-cols-1 lg:grid-cols-3 lg:divide-x divide-slate-100"
                    ):
                        with ui.column().classes("p-5 gap-2"):
                            ui.label("Run").classes("text-xs font-bold text-slate-400")
                            _field("Run ID", str(meta.get("run_id", "—")), mono=True)
                            _field("Seed", str(meta.get("seed", cfg.get("seed", "—"))))
                            _field("Iterations logged", str(len(df)))

                        with ui.column().classes("p-5 gap-2 lg:border-x border-slate-100"):
                            ui.label("Progress").classes("text-xs font-bold text-slate-400")
                            _field("Iteration", f"{cur} / {total or '—'}")
                            ui.linear_progress(value=min(cur / max(total, 1), 1.0)).classes("w-full")
                            if is_cyclic:
                                cycles = plan.get("cycles", 5)
                                ui.label(f"Cyclic: C → R → RC × {cycles}").classes(
                                    "text-sm font-medium text-slate-700"
                                )
                            phase_label = status.get("training_phase_name") or status.get("stage_name")
                            if phase_label:
                                st = int(status.get("curriculum_stage") or 0)
                                ui.label(phase_label).classes("text-sm font-medium").style(
                                    f"color:{STAGE_COLORS.get(st, '#2563eb')}"
                                )
                            if not df.empty and "instance_name" in df.columns:
                                last_inst = df.iloc[-1].get("instance_name")
                                if last_inst:
                                    _field("Last instance", str(last_inst), mono=True)
                            if status.get("message"):
                                ui.label(status["message"]).classes("text-xs text-slate-400")

                        with ui.column().classes("p-5 gap-2"):
                            ui.label("Schedule").classes("text-xs font-bold text-slate-400")
                            if is_cyclic:
                                order = " → ".join(plan.get("class_order") or ["C", "R", "RC"])
                                ui.label(f"{order} · 1 instance = 1 iteration").classes(
                                    "text-xs text-slate-500"
                                )
                            _field("Algorithm", "REINFORCE + greedy baseline")
                            ui.label("outputs/results/training_metrics.csv").classes(
                                "text-[10px] font-mono text-slate-400 mt-1"
                            )

            # ---- Reward KPIs (top, right after status) ----
            reward_kpi_panel.clear()
            if not df.empty:
                last = df.iloc[-1]

                def pick(roll: str, raw: str):
                    if roll in df.columns:
                        v = last.get(roll)
                        if v is not None and v == v:
                            return v
                    return last.get(raw)

                with reward_kpi_panel:
                    with ui.card().classes(_SECTION):
                        ui.label("Reward & optimization · latest step").classes(
                            "text-xs font-bold text-slate-400 px-5 pt-4"
                        )
                        with ui.row().classes("w-full flex-wrap justify-around pb-2"):
                            _kpi("Reward", _fmt_num(pick("rolling_raw_return_mean", "reward"), 2))
                            _kpi("Cost J", _fmt_num(last.get("cost"), 1))
                            _kpi("Loss", _fmt_num(last.get("loss"), 3))
                            _kpi("Policy loss", _fmt_num(last.get("policy_loss"), 3))
                            _kpi("Advantage", _fmt_num(last.get("advantage"), 3))
                            _kpi("Entropy", _fmt_num(last.get("entropy"), 3))
                            _kpi("Grad norm", _fmt_num(last.get("grad_norm"), 3))
                            _kpi("LR", _fmt_num(last.get("learning_rate"), 6))
                            _kpi("Served", _fmt_pct(pick("rolling_served_ratio_mean", "served_ratio")))

                # Per-class snapshot (trailing window)
                if "instance_class" in df.columns:
                    w = min(10, max(3, len(df) // 3))
                    with reward_kpi_panel:
                        with ui.card().classes(_SECTION):
                            ui.label("Per-class snapshot · trailing window mean").classes(
                                "text-xs font-bold text-slate-400 px-5 pt-4"
                            )
                            with ui.row().classes("w-full flex-wrap gap-3 px-4 pb-4"):
                                for cls in CLASS_ORDER:
                                    g = df[df["instance_class"] == cls].tail(w)
                                    if g.empty:
                                        continue
                                    color = CLASS_COLORS.get(cls, "#64748b")
                                    with ui.column().classes(
                                        "flex-1 min-w-[10rem] rounded-lg border border-slate-100 p-3"
                                    ).style(f"border-top: 3px solid {color}"):
                                        ui.label(cls).classes("text-sm font-bold").style(
                                            f"color:{color}"
                                        )
                                        ui.label(f"Reward {_fmt_num(g['reward'].mean(), 2)}").classes(
                                            "text-xs text-slate-600"
                                        )
                                        if "cost" in g.columns:
                                            ui.label(f"Cost J {_fmt_num(g['cost'].mean(), 1)}").classes(
                                                "text-xs text-slate-600"
                                            )
                                        if "served_ratio" in g.columns:
                                            ui.label(
                                                f"Served {_fmt_pct(g['served_ratio'].mean())}"
                                            ).classes("text-xs text-slate-600")
                                        if "avg_used_evs" in g.columns:
                                            ui.label(
                                                f"EVs {_fmt_num(g['avg_used_evs'].mean(), 1)}"
                                            ).classes("text-xs text-slate-600")

            # ---- Charts (flat sections, no dropdowns) ----
            charts_panel.clear()
            with charts_panel:
                if df.empty:
                    with ui.card().classes(_SECTION + " p-10 text-center"):
                        ui.label("Waiting for metrics…").classes("text-slate-600")
                        ui.label(
                            "python main_train.py --config configs/curriculum.yaml --curriculum"
                        ).classes("text-xs font-mono text-slate-400 mt-2")
                    return

                for n, section in enumerate(TRAINING_CHART_SECTIONS, 1):
                    charts = section.get("charts", [])
                    if not _section_has_data(df, charts):
                        continue
                    accent = _ACCENTS[(n - 1) % len(_ACCENTS)]
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
                                    ui.label(section["subtitle"]).classes("text-xs text-slate-500")
                        with ui.column().classes(_SECTION_BODY).style("width: 100%;"):
                            _render_charts(df, plan, charts)

        with ui.header().classes("bg-white border-b border-slate-200 text-slate-800"):
            with ui.row().classes(
                "w-full max-w-[1280px] mx-auto px-6 py-3 items-center justify-between"
            ):
                with ui.column().classes("gap-0"):
                    ui.label("E-VRPTW Transformer · Training monitor").classes("text-lg font-bold")
                    ui.label(str(log_path)).classes("text-xs text-slate-400")
                with ui.row().classes("gap-3 items-center"):
                    ui.button("Refresh", on_click=refresh, icon="refresh").props("outline dense")
                    auto = ui.switch("Auto-refresh (3s)", value=True).props("dense")
            ui.label(
                "Reward metrics first · C → R → RC cyclic schedule · 1 instance = 1 iteration"
            ).classes("text-[11px] text-slate-400 px-6 pb-2 w-full max-w-[1280px] mx-auto block")

        poll = ui.timer(3.0, refresh)
        poll.active = True
        auto.on("update:model-value", lambda e: setattr(poll, "active", bool(e.value)))
        ui.timer(0.1, refresh, once=True)
