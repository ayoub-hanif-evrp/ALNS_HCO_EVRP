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
from src.webapp.plot_style import STAGE_COLORS

_PAGE = "w-full max-w-[1400px] mx-auto px-6"
_SECTION = "w-full rounded-xl border border-slate-200 bg-white shadow-sm mb-5"
_SECTION_HEAD = "px-5 py-3 border-b border-slate-100 bg-slate-50"
_SECTION_BODY = "p-4"
_TILE = "rounded-lg border border-slate-100 bg-white p-3"
_ACCENTS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5"]


def _field(label: str, value: str, *, mono: bool = False) -> None:
    ui.label(label).classes("text-[11px] uppercase tracking-wide text-slate-400")
    ui.label(value).classes(
        "text-sm font-semibold text-slate-800" + (" font-mono" if mono else "")
    )


def _section_has_data(df, charts: list) -> bool:
    for c in charts:
        t = c.get("type", "metric")
        if t in ("cost_breakdown", "reward_components", "served_coverage", "optimization_panel", "cost_term"):
            if t == "cost_term" and c.get("col") in df.columns:
                return True
            if t != "cost_term":
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
    desc = spec.get("desc") if spec.get("type") != "cost_term" else None
    with ui.element("div").classes("chart-slot").style("width: 100%; min-width: 0;"):
        with ui.column().classes(_TILE).style("width: 100%; box-sizing: border-box;"):
            ui.label(title).classes("text-sm font-semibold text-slate-800 mb-2")
            if desc:
                ui.label(desc).classes("text-xs text-slate-400 mb-2")
            ui.plotly(_plotly_payload(fig)).classes("w-full").style(
                "width: 100%; display: block; min-width: 0;"
            )


def _render_section_card(
    section: dict,
    df,
    plan: dict,
    *,
    accent: str,
) -> None:
    """One themed block: outer div + white card header + chart grid (same pattern as Policy optimization)."""
    section_id = section.get("id", "section")
    with ui.element("div").classes("dashboard-section w-full scroll-mt-24").props(
        f'id="section-{section_id}"'
    ):
        with ui.card().classes(_SECTION + " section-card"):
            with ui.row().classes(_SECTION_HEAD + " items-center gap-3"):
                ui.element("div").classes("w-1.5 h-9 rounded-full shrink-0").style(
                    f"background:{accent}"
                )
                with ui.column().classes("gap-0"):
                    ui.label(section["title"]).classes("text-base font-semibold text-slate-800")
                    if section.get("subtitle"):
                        ui.label(section["subtitle"]).classes("text-xs text-slate-500")
            with ui.column().classes(_SECTION_BODY).style("width: 100%;"):
                _render_charts(df, plan, section.get("charts", []))


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


def _section_nav_link(section: dict) -> None:
    sid = section.get("id", "")
    label = section.get("nav") or section.get("title", sid)
    ui.link(label, f"#section-{sid}").classes(
        "text-xs font-medium text-slate-600 hover:text-blue-600 px-2.5 py-1 rounded-full "
        "border border-slate-200 bg-white hover:border-blue-200 whitespace-nowrap"
    )


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
              .dashboard-section { margin-bottom: 1.25rem; }
              .dashboard-section .section-card { margin-bottom: 0 !important; }
              .section-nav { position: sticky; top: 0; z-index: 20; background: #f1f5f9;
                             padding: 0.75rem 0; margin-bottom: 0.5rem; }
            </style>
            """
        )

        status_panel = ui.column().classes(_PAGE + " pt-4")
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
                            _field("Algorithm", "REINFORCE + baseline")
                            ui.label("outputs/results/training_metrics.csv").classes(
                                "text-[10px] font-mono text-slate-400 mt-1"
                            )

            # ---- Charts ----
            charts_panel.clear()
            with charts_panel:
                if df.empty:
                    with ui.card().classes(_SECTION + " p-10 text-center"):
                        ui.label("Waiting for metrics…").classes("text-slate-600")
                        ui.label(
                            "python main_train.py --config configs/curriculum.yaml --curriculum"
                        ).classes("text-xs font-mono text-slate-400 mt-2")
                    return

                visible_sections = [
                    s for s in TRAINING_CHART_SECTIONS
                    if _section_has_data(df, s.get("charts", []))
                ]

                with ui.element("div").classes("section-nav w-full"):
                    ui.label("Sections").classes("text-[10px] uppercase tracking-wide text-slate-400 mb-2")
                    with ui.row().classes("flex-wrap gap-2 w-full"):
                        for section in visible_sections:
                            _section_nav_link(section)

                for n, section in enumerate(visible_sections, 1):
                    accent = _ACCENTS[(n - 1) % len(_ACCENTS)]
                    _render_section_card(section, df, plan, accent=accent)

        with ui.header().classes("bg-white border-b border-slate-200 text-slate-800"):
            with ui.row().classes(
                "w-full max-w-[1400px] mx-auto px-6 py-3 items-center justify-between"
            ):
                with ui.column().classes("gap-0"):
                    ui.label("E-VRPTW Transformer · Training monitor").classes("text-lg font-bold")
                    ui.label(str(log_path)).classes("text-xs text-slate-400")
                with ui.row().classes("gap-3 items-center"):
                    ui.button("Refresh", on_click=refresh, icon="refresh").props("outline dense")
                    auto = ui.switch("Auto-refresh (3s)", value=True).props("dense")
            ui.label(
                "Grouped sections · one card per theme · jump links above charts"
            ).classes("text-[11px] text-slate-400 px-6 pb-2 w-full max-w-[1400px] mx-auto block")

        poll = ui.timer(3.0, refresh)
        poll.active = True
        auto.on("update:model-value", lambda e: setattr(poll, "active", bool(e.value)))
        ui.timer(0.1, refresh, once=True)
