"""Maps / solution-behavior explorer — manual instance & epoch selection (no auto-refresh)."""

from __future__ import annotations

import asyncio
import base64
from functools import partial
from pathlib import Path

import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

from src.webapp.dashboard_data import filter_training_instances, load_metrics_df, load_run_config
from src.webapp.maps_dashboard_data import (
    generate_instance_snapshot,
    instance_epoch_rows,
    instance_progression_records,
    list_training_instances,
    maps_figures_root,
    maps_traces_root,
    snapshot_dir,
    snapshot_is_cached,
)
from src.webapp.plot_style import OVERALL_LINE_COLOR

_PAGE = "w-full max-w-[1400px] mx-auto px-6"
_SECTION = "w-full rounded-xl border border-slate-200 bg-white shadow-sm mb-5"
_TILE = "rounded-lg border border-slate-100 bg-white p-3"


def _progression_figure(records: list[dict]) -> go.Figure | None:
    if not records:
        return None
    epochs = [r["epoch"] for r in records]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=epochs, y=[r["served_rate"] for r in records], name="Served ratio",
                   line=dict(color="#059669", width=2), mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=[r["objective"] for r in records], name="Objective J",
                   line=dict(color=OVERALL_LINE_COLOR, width=2), mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        autosize=True,
        margin=dict(t=24, b=40, l=52, r=52),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=None,
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Served ratio", secondary_y=False)
    fig.update_yaxes(title_text="Objective J", secondary_y=True)
    return fig


def _plotly_payload(fig) -> dict:
    payload = fig.to_plotly_json()
    payload.setdefault("layout", {})["autosize"] = True
    payload.setdefault("config", {})["responsive"] = True
    payload.setdefault("config", {})["displayModeBar"] = False
    return payload


def _png_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_maps_dashboard(
    log_path: str | Path,
    *,
    config_path: str | Path | None = None,
    outputs_root: str | Path | None = None,
    main_dashboard_port: int = 8080,
) -> None:
    log_path = Path(log_path)
    config_path = Path(config_path) if config_path else None
    outputs_root = Path(outputs_root) if outputs_root else Path("outputs")

    @ui.page("/")
    def page() -> None:  # noqa: ANN202
        ui.colors(primary="#2563eb")
        ui.add_head_html(
            """
            <style>
              body { background: #f1f5f9 !important; }
              .nicegui-content { padding: 0 !important; }
              .diag-grid img { width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; }
              .map-hero img {
                width: 100%;
                max-height: none;
                border-radius: 10px;
                border: 2px solid #cbd5e1;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
                background: #fff;
              }
              .map-hero { width: 100%; }
            </style>
            """
        )

        state: dict = {"df": None, "cfg": {}, "snapshot": None, "selected_epoch": None}

        with ui.header().classes("bg-white border-b border-slate-200 text-slate-800"):
            with ui.row().classes("w-full max-w-[1400px] mx-auto px-6 py-3 items-center justify-between"):
                with ui.column().classes("gap-0"):
                    ui.label("MARL EVRP · Solution behavior maps").classes("text-lg font-bold")
                    ui.label("Select instance & epoch — no auto-refresh").classes("text-xs text-slate-400")
                ui.link(
                    "← Training metrics dashboard",
                    f"http://127.0.0.1:{main_dashboard_port}",
                ).classes("text-xs text-blue-600")

        body = ui.column().classes(_PAGE + " py-4 pb-16 gap-4")
        progression_panel: ui.column
        meta_panel: ui.column
        diag_panel: ui.column
        instance_select: ui.select
        epoch_select: ui.select
        force_cb: ui.checkbox
        status_label: ui.label
        load_btn: ui.button

        def _load_metrics() -> bool:
            cfg = load_run_config(outputs_root, config_path)
            raw = load_metrics_df(log_path)
            df = filter_training_instances(raw, cfg)
            state["cfg"] = cfg
            state["df"] = df
            maps_figures_root(cfg, outputs_root).mkdir(parents=True, exist_ok=True)
            return not df.empty

        def _epoch_options(instance_name: str) -> dict[str, int]:
            sub = instance_epoch_rows(state["df"], instance_name)
            if sub.empty:
                return {}
            fig_root = maps_figures_root(state["cfg"], outputs_root)
            opts = {}
            for _, row in sub.iterrows():
                ep = int(row["epoch"])
                cyc = int(row.get("cycle", 0))
                sr = float(row.get("served_rate", row.get("served_ratio", 0)) or 0)
                obj = float(row.get("final_objective", row.get("cost", 0)) or 0)
                cached_mark = "✓ cached" if snapshot_is_cached(snapshot_dir(fig_root, instance_name, ep)) else ""
                label = f"epoch {ep} · cycle {cyc} · served {sr:.0%} · J={obj:.0f}"
                if cached_mark:
                    label += f" · {cached_mark}"
                opts[label] = ep
            return opts

        def _update_progression(instance_name: str | None, highlight_epoch: int | None = None) -> None:
            progression_panel.clear()
            if not instance_name:
                return
            records = instance_progression_records(state["df"], instance_name)
            fig = _progression_figure(records)
            with progression_panel:
                ui.label(f"Training progression · {instance_name}").classes(
                    "text-sm font-semibold text-slate-800 mb-2"
                )
                if fig is None:
                    ui.label("No logged epochs for this instance.").classes("text-xs text-slate-400")
                    return
                if highlight_epoch is not None:
                    fig.add_vline(x=highlight_epoch, line_dash="dot", line_color="#dc2626", line_width=2)
                ui.plotly(_plotly_payload(fig)).classes("w-full")

        def _selected_epoch() -> int | None:
            ep = epoch_select.value
            if ep is None:
                return None
            if isinstance(ep, (int, float)):
                return int(ep)
            opts = epoch_select.options or {}
            if isinstance(opts, dict):
                if ep in opts:
                    return int(opts[ep])
                for label, value in opts.items():
                    if str(value) == str(ep) or str(label) == str(ep):
                        return int(value)
            try:
                return int(ep)
            except (TypeError, ValueError):
                return None

        def _on_instance_change() -> None:
            name = instance_select.value
            if not name:
                return
            opts = _epoch_options(name)
            epoch_select.options = opts
            epoch_select.value = list(opts.values())[-1] if opts else None
            state["selected_epoch"] = epoch_select.value
            _update_progression(name, epoch_select.value)

        def _on_epoch_change() -> None:
            state["selected_epoch"] = epoch_select.value
            _update_progression(instance_select.value, epoch_select.value)

        def _render_snapshot(meta: dict | None) -> int:
            meta_panel.clear()
            diag_panel.clear()
            if not meta or not meta.get("ok"):
                with meta_panel:
                    msg = (meta or {}).get("error", "Select an instance and epoch, then click Load snapshot.")
                    ui.label(msg).classes("text-sm text-rose-600" if meta else "text-sm text-slate-500")
                return 0

            with meta_panel:
                with ui.card().classes(_SECTION + " p-4"):
                    ui.label("Snapshot summary").classes("text-sm font-semibold text-slate-800 mb-2")
                    ui.label(
                        f"{meta['instance_name']} · logged epoch {meta['epoch']} · cycle {meta.get('cycle', '?')} · "
                        f"J={meta['objective']:.0f} · served {meta['served_ratio']:.1%} · "
                        f"feasible={meta['feasible']} · unserved={meta['unserved_customers']} · "
                        f"EVs={meta['used_evs']} · dist={meta.get('total_distance', 0):.0f}"
                    ).classes("text-xs font-mono text-slate-700")
                    ui.label(
                        f"Checkpoint: {meta['checkpoint']} · {meta.get('checkpoint_note', '')}"
                        + (" · from cache" if meta.get("cached") else "")
                    ).classes("text-xs text-slate-500 mt-1")

            sections = [
                (
                    "EV route timeline",
                    ["live_route_timeline.png"],
                    "How each EV spends time: travel, service, charging, depot returns.",
                ),
                (
                    "Battery trajectory",
                    ["live_battery_trajectory.png"],
                    "Battery level over time. Jumps = charging. Red zone = below zero.",
                ),
                (
                    "Unserved customers",
                    ["live_unserved_over_decisions.png"],
                    "Remaining unserved, expired TW, and still-feasible customers per decision.",
                ),
                (
                    "Action sequence",
                    ["live_action_sequence.png"],
                    "What the policy chose at each step — customers vs stations vs depot.",
                ),
                (
                    "Feasible vs masked actions",
                    ["live_feasible_actions.png"],
                    "Top: available actions. Bottom: customers blocked by TW, battery, capacity.",
                ),
                (
                    "Time-window slack",
                    ["live_service_slack.png"],
                    "How early/late each customer was served relative to due time.",
                ),
                (
                    "Per-EV workload",
                    ["live_ev_route_balance.png"],
                    "Customers, distance, charging time, and duration per EV.",
                ),
            ]
            map_file = "live_solution_map.png"
            figures_dir = Path(meta.get("figures_dir", ""))
            shown = 0

            map_meta = next((f for f in meta.get("figures", []) if f["file"] == map_file), None)
            if map_meta and map_meta.get("exists"):
                map_path = figures_dir / map_file
                if map_path.is_file():
                    shown += 1
                    with diag_panel:
                        with ui.card().classes(_SECTION + " p-4 map-hero"):
                            ui.label("Full EVRP solution map").classes(
                                "text-base font-bold text-slate-800 mb-1"
                            )
                            ui.label(
                                "Depot, customers (served/unserved), stations, EV routes with "
                                "visit order, travel arrows, depot returns, and route summary panel."
                            ).classes("text-xs text-slate-500 mb-3")
                            ui.image(_png_data_url(map_path)).classes("w-full map-hero")

            with diag_panel:
                with ui.column().classes("w-full gap-4"):
                    for title, files, desc in sections:
                        fig_meta = next((f for f in meta.get("figures", []) if f["file"] in files), None)
                        if not fig_meta or not fig_meta.get("exists"):
                            continue
                        img_path = figures_dir / fig_meta["file"]
                        if not img_path.is_file():
                            continue
                        shown += 1
                        with ui.card().classes(_SECTION + " p-4"):
                            ui.label(title).classes("text-sm font-bold text-slate-800 mb-1")
                            ui.label(desc).classes("text-xs text-slate-500 mb-3")
                            ui.image(_png_data_url(img_path)).classes("w-full diag-grid")
                if shown == 0:
                    ui.label(
                        "No PNG diagnostics found on disk for this snapshot. "
                        "Try “Regenerate (ignore cache)” and load again."
                    ).classes("text-sm text-amber-700")
            return shown

        def _step_epoch(delta: int) -> None:
            if not epoch_select.options:
                return
            values = list(epoch_select.options.values())
            if epoch_select.value not in values:
                return
            idx = values.index(epoch_select.value)
            idx = max(0, min(len(values) - 1, idx + delta))
            epoch_select.value = values[idx]
            state["selected_epoch"] = values[idx]
            _update_progression(instance_select.value, values[idx])

        async def _load_snapshot() -> None:
            inst = instance_select.value
            ep = _selected_epoch()
            if not inst or ep is None:
                status_label.text = "Choose an instance and epoch first."
                return
            status_label.text = "Generating snapshot…"
            load_btn.props("loading")
            try:
                run_snapshot = partial(
                    generate_instance_snapshot,
                    state["cfg"],
                    state["df"],
                    inst,
                    ep,
                    figures_root=maps_figures_root(state["cfg"], outputs_root),
                    traces_root=maps_traces_root(state["cfg"], outputs_root),
                    force=bool(force_cb.value),
                )
                meta = await asyncio.to_thread(run_snapshot)
                state["snapshot"] = meta
                n_figs = _render_snapshot(meta)
                if meta.get("ok"):
                    status_label.text = f"Snapshot loaded — {n_figs} diagnostic figure(s). Scroll down to view."
                else:
                    status_label.text = meta.get("error", "Failed")
                # refresh epoch labels for cache marks
                epoch_select.options = _epoch_options(inst)
                epoch_select.value = ep
            except Exception as exc:
                err = str(exc)
                status_label.text = f"Snapshot failed: {err}"
                _render_snapshot({"ok": False, "error": err})
            finally:
                load_btn.props(remove="loading")

        def _reload_metrics() -> None:
            ok = _load_metrics()
            if not ok:
                status_label.text = "No training metrics yet — run main_train.py first."
                return
            instances = list_training_instances(state["df"])
            instance_select.options = instances
            instance_select.value = instances[-1] if instances else None
            if instance_select.value:
                epoch_select.options = _epoch_options(instance_select.value)
                epoch_select.value = list(epoch_select.options.values())[-1] if epoch_select.options else None
                _update_progression(instance_select.value, epoch_select.value)
            status_label.text = f"Loaded {len(state['df'])} logged rows · {len(instances)} instances."

        with body:
            with ui.card().classes(_SECTION + " p-4"):
                ui.label("Explore solution behavior").classes("text-base font-semibold text-slate-800 mb-3")
                with ui.row().classes("flex-wrap gap-4 items-end w-full"):
                    instance_select = ui.select([], label="Training instance").classes("min-w-[220px]")
                    epoch_select = ui.select([], label="Logged epoch").classes("min-w-[280px]")
                    with ui.row().classes("gap-2"):
                        ui.button("◀ Prev epoch", on_click=lambda: _step_epoch(-1)).props("outline dense")
                        ui.button("Next epoch ▶", on_click=lambda: _step_epoch(1)).props("outline dense")
                    load_btn = ui.button("Load snapshot", on_click=_load_snapshot, icon="map").props("color=primary")
                    ui.button("Reload training log", on_click=_reload_metrics, icon="refresh").props("outline dense")
                    force_cb = ui.checkbox("Regenerate (ignore cache)", value=False)
                status_label = ui.label("").classes("text-xs text-slate-500")

            with ui.card().classes(_SECTION + " p-4"):
                progression_panel = ui.column().classes("w-full")

            meta_panel = ui.column().classes("w-full")
            diag_panel = ui.column().classes("w-full gap-4")

        instance_select.on_value_change(_on_instance_change)
        epoch_select.on_value_change(_on_epoch_change)

        ui.timer(0.05, _reload_metrics, once=True)

    return None
