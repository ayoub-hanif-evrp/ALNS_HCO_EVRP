"""Shared Plotly styling for the training dashboard."""

from __future__ import annotations

# Calm, muted, scientific palette.
COLORS = {
    "blue": "#2563eb",
    "green": "#16a34a",
    "teal": "#0f766e",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#64748b",
    "muted": "#94a3b8",
    "gold": "#ca8a04",
    "rose": "#e11d48",
    "navy": "#1e293b",
}

SEQ = [
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0f766e",
    "#dc2626",
    "#ca8a04",
    "#64748b",
    "#e11d48",
    "#0891b2",
]

SECTION_ACCENTS = [
    "#2563eb",
    "#7c3aed",
    "#059669",
    "#d97706",
    "#dc2626",
    "#0891b2",
    "#4f46e5",
]


def base_layout(*, ylabel: str, xlabel: str = "epoch", showlegend: bool | None = None) -> dict:
    layout = dict(
        template="simple_white",
        autosize=True,
        height=320,
        margin=dict(l=56, r=20, t=8, b=40),
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12, color="#334155"),
        xaxis=dict(
            title=xlabel,
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            ticks="outside",
            tickcolor="#cbd5e1",
        ),
        yaxis=dict(
            title=ylabel,
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            ticks="outside",
            tickcolor="#cbd5e1",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    if showlegend is not None:
        layout["showlegend"] = showlegend
    return layout
