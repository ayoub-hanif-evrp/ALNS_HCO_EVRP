"""Quality tests for main-paper EVRPTW figures."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "src" / "visualization" / "final_paper_figures.py"
PAPER = ROOT / "outputs"
MAIN = PAPER / "figures"
META = PAPER / "manuscript_values.json"
SEL = PAPER / "representative_routes.json"

FORBIDDEN = ("hanif", "warm start", "warm-start", "seed", "debug")


def test_generator_has_no_dashboard_grids():
    text = GEN.read_text(encoding="utf-8")
    assert "subplots(2, 3" not in text
    assert "subplots(2, 2" not in text
    assert "GridSpec" not in text


def test_generator_titles_are_scientific():
    titles = [
        line for line in GEN.read_text(encoding="utf-8").splitlines()
        if "set_title(" in line
    ]
    blob = "\n".join(titles).lower()
    for bad in ("hanif", "warm start", "seed", "debug", "det-fps", "a->b"):
        assert bad not in blob


def test_main_figure_artifacts_if_present():
    if not MAIN.is_dir():
        return
    stems = sorted({p.stem for p in MAIN.iterdir() if p.suffix in {".png", ".svg"}})
    assert len(stems) <= 6
    for stem in stems:
        assert (MAIN / f"{stem}.svg").is_file()
        assert (MAIN / f"{stem}.png").is_file()
    if META.is_file():
        import json
        meta = json.loads(META.read_text(encoding="utf-8"))
        for title in meta.get("titles") or []:
            low = title.lower()
            for bad in FORBIDDEN:
                assert bad not in low
        assert int(meta.get("n_main_figures") or 0) <= 6
    if SEL.is_file():
        import json
        sel = json.loads(SEL.read_text(encoding="utf-8"))
        # Legacy paper/ artifacts may still list 5 routes; the generator now uses 3.
        assert len(sel.get("highlighted_routes") or []) <= 5
        assert len(sel.get("soc_routes") or []) <= 2
