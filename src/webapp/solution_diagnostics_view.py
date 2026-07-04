"""NiceGUI rendering for solution-behavior diagnostic figures."""

from __future__ import annotations

from pathlib import Path

from nicegui import app, ui

from src.webapp.solution_diagnostics_data import (
    FIGURES_MOUNT,
    TRACES_MOUNT,
    build_solution_behavior_section,
    figures_dir_from_config,
    traces_dir_from_config,
)

_MOUNTED_FIGURES: Path | None = None
_MOUNTED_TRACES: Path | None = None

_TILE = "rounded-lg border border-slate-100 bg-white p-3"
_TABLE_STYLE = (
    "<style>.diag-table{border-collapse:collapse;width:100%;font-size:12px}"
    "th,td{border:1px solid #cbd5e1;padding:4px 8px;text-align:left}"
    "th{background:#f1f5f9;color:#334155;font-weight:600}"
    "td{color:#334155}</style>"
)


def mount_solution_static_assets(figures_dir: Path, traces_dir: Path | None = None) -> tuple[str, str]:
    """Register NiceGUI static mounts for PNG/HTML figures and JSON traces."""
    global _MOUNTED_FIGURES, _MOUNTED_TRACES
    figures_dir = figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    if _MOUNTED_FIGURES != figures_dir:
        try:
            app.add_static_files(FIGURES_MOUNT, str(figures_dir))
        except ValueError:
            pass
        _MOUNTED_FIGURES = figures_dir

    if traces_dir is not None:
        traces_dir = traces_dir.resolve()
        traces_dir.mkdir(parents=True, exist_ok=True)
        if _MOUNTED_TRACES != traces_dir:
            try:
                app.add_static_files(TRACES_MOUNT, str(traces_dir))
            except ValueError:
                pass
            _MOUNTED_TRACES = traces_dir

    return FIGURES_MOUNT, TRACES_MOUNT


def render_solution_behavior_section(section: dict, *, accent: str) -> None:
    """Render solution-behavior diagnostics inside a dashboard section card."""
    section_id = section.get("id", "solution_behavior")
    with ui.element("div").classes("dashboard-section w-full scroll-mt-24").props(
        f'id="section-{section_id}"'
    ):
        with ui.card().classes(
            "w-full rounded-xl border border-slate-200 bg-white shadow-sm mb-5 section-card"
        ):
            with ui.row().classes(
                "px-5 py-3 border-b border-slate-100 bg-slate-50 items-center gap-3"
            ):
                ui.element("div").classes("w-1.5 h-9 rounded-full shrink-0").style(
                    f"background:{accent}"
                )
                with ui.column().classes("gap-0"):
                    ui.label(section["title"]).classes("text-base font-semibold text-slate-800")
                    if section.get("subtitle"):
                        ui.label(section["subtitle"]).classes("text-xs text-slate-500")

            with ui.column().classes("p-4 gap-6").style("width: 100%;"):
                live = section.get("live") or {}
                live_state = live.get("state") or {}
                if live.get("enabled"):
                    with ui.column().classes(
                        "gap-2 rounded-lg border border-blue-100 bg-blue-50/60 px-3 py-2"
                    ):
                        if live.get("generating"):
                            ui.label("Generating live snapshot from latest checkpoint…").classes(
                                "text-xs text-blue-800"
                            )
                        elif live_state.get("instance_name"):
                            age = live.get("age_sec")
                            age_txt = f"{age}s ago" if age is not None else "just now"
                            ui.label(
                                f"Live snapshot · {live_state['instance_name']} · "
                                f"epoch {live_state.get('epoch', '?')} · "
                                f"J={float(live_state.get('objective', 0)):.0f} · "
                                f"served {float(live_state.get('served_ratio', 0)):.1%} · "
                                f"feasible={live_state.get('feasible')} · "
                                f"checkpoint {live_state.get('checkpoint', '?')} · "
                                f"refreshed {age_txt}"
                            ).classes("text-xs text-blue-900 font-mono")
                        elif live.get("error") or live_state.get("error"):
                            ui.label(str(live.get("error") or live_state.get("error"))).classes(
                                "text-xs text-amber-800"
                            )
                        else:
                            ui.label(
                                "Waiting for first live snapshot (needs a saved checkpoint)."
                            ).classes("text-xs text-blue-800")

                    for group in live.get("groups") or []:
                        ui.label(group["title"]).classes(
                            "text-sm font-semibold text-slate-700 border-b border-slate-100 pb-1"
                        )
                        with ui.element("div").classes("grid grid-cols-1 xl:grid-cols-2 gap-4 w-full"):
                            for item in group.get("items") or []:
                                with ui.column().classes(_TILE).style(
                                    "width: 100%; box-sizing: border-box;"
                                ):
                                    ui.label(item["title"]).classes(
                                        "text-sm font-semibold text-slate-800 mb-2"
                                    )
                                    if item.get("exists"):
                                        ui.image(
                                            f"{FIGURES_MOUNT}/{item['file']}?v={item['mtime']}"
                                        ).classes("w-full rounded border border-slate-100")
                                    elif live.get("generating"):
                                        ui.label("Generating…").classes(
                                            "text-xs text-slate-500 py-6 text-center"
                                        )
                                    else:
                                        ui.label("Waiting for first live snapshot.").classes(
                                            "text-xs text-slate-400 py-6 text-center"
                                        )

                static_groups = section.get("groups") or []
                if any(
                    item.get("exists")
                    for group in static_groups
                    for item in group.get("items") or []
                ):
                    ui.label("Post-training representative episodes").classes(
                        "text-sm font-semibold text-slate-700 border-b border-slate-100 pb-1 mt-2"
                    )
                elif section.get("figures_pending") and not live.get("enabled"):
                    ui.label(
                        "Route-level figures appear when training finishes "
                        "(dashboard.save_representative_traces: true)."
                    ).classes("text-xs text-amber-700 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2")

                for group in static_groups:
                    visible = [i for i in group.get("items") or [] if i.get("exists")]
                    if not visible:
                        continue
                    ui.label(group["title"]).classes(
                        "text-sm font-semibold text-slate-700 border-b border-slate-100 pb-1"
                    )
                    with ui.element("div").classes("grid grid-cols-1 xl:grid-cols-2 gap-4 w-full"):
                        for item in visible:
                            with ui.column().classes(_TILE).style("width: 100%; box-sizing: border-box;"):
                                req = " · critical" if item.get("required") else ""
                                ui.label(f"{item['title']}{req}").classes(
                                    "text-sm font-semibold text-slate-800 mb-2"
                                )
                                ui.image(
                                    f"{FIGURES_MOUNT}/{item['file']}?v={item['mtime']}"
                                ).classes("w-full rounded border border-slate-100")

                if section.get("table_html"):
                    ui.label("Final-cycle instance table").classes(
                        "text-sm font-semibold text-slate-700 border-b border-slate-100 pb-1"
                    )
                    ui.html(_TABLE_STYLE + section["table_html"]).classes(
                        "w-full overflow-x-auto"
                    )

                traces = section.get("trace_files") or []
                if traces:
                    ui.label("Representative rollout traces").classes(
                        "text-sm font-semibold text-slate-700 border-b border-slate-100 pb-1"
                    )
                    with ui.row().classes("flex-wrap gap-2"):
                        for tr in traces:
                            ui.link(tr["name"], tr["url"]).classes(
                                "text-xs font-mono text-blue-600 hover:underline"
                            ).props("target=_blank")

                if section.get("static_index_url"):
                    ui.link(
                        "Open full static training dashboard (index.html)",
                        section["static_index_url"],
                    ).classes("text-xs text-blue-600 hover:underline").props("target=_blank")


__all__ = [
    "build_solution_behavior_section",
    "figures_dir_from_config",
    "traces_dir_from_config",
    "mount_solution_static_assets",
    "render_solution_behavior_section",
]
