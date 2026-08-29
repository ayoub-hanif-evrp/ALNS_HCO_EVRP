"""Paper-readiness helpers: protocols, completeness, literature competitors."""
from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.hanif_protocols import (
    PROTOCOL_EXTENDED,
    PROTOCOL_NO_WAIT_REPLAY,
    PROTOCOL_PROPOSED_EXTENDED,
    completeness_report,
    instance_without_station_wait,
    protocols_compatible,
)
from src.evaluation.literature_competitors import competitor_table_rows


def test_protocol_names_are_explicit():
    assert PROTOCOL_EXTENDED["name"] == "EXTENDED"
    assert PROTOCOL_NO_WAIT_REPLAY["name"] == "NO_WAIT_REPLAY"
    assert PROTOCOL_NO_WAIT_REPLAY["reoptimizes"] is False
    assert PROTOCOL_EXTENDED["charging_price_in_objective"] is False
    assert PROTOCOL_PROPOSED_EXTENDED["charging_price_in_objective"] is True
    assert protocols_compatible("EXTENDED", "HANIF_EXTENDED_DETERMINISTIC")
    assert not protocols_compatible("EXTENDED", "NO_WAIT_REPLAY")
    assert protocols_compatible("EXTENDED", None)


def test_completeness_refuses_partial_means():
    report = completeness_report(
        expected_names=["C1_10_1", "R1_10_1"],
        got_names=["C1_10_1"],
        failures=[{"instance": "R1_10_1", "error": "boom"}],
    )
    assert report["complete"] is False
    assert report["cite_means"] is False
    assert report["missing_instances"] == ["R1_10_1"]


def test_completeness_ok_when_all_present():
    report = completeness_report(
        expected_names=["C1_10_1", "R1_10_1"],
        got_names=["R1_10_1", "C1_10_1"],
    )
    assert report["complete"] is True
    assert report["cite_means"] is True


def test_literature_competitors_include_alns_and_vns():
    rows = competitor_table_rows()
    ids = {r["method_id"] for r in rows}
    assert "ALNS" in ids
    assert "VNS_TS" in ids
    assert "ILS" in ids
    assert any(r["reimplemented_on_this_benchmark"] for r in rows)
    assert any(not r["reimplemented_on_this_benchmark"] for r in rows)


def test_no_wait_clone_zeros_wait_mu(tmp_path: Path):
    hanif = Path("dataset/Hanif/converted_hanif_original_tw/C1_10_1")
    if not (hanif / "evrp_instance.json").is_file():
        import pytest
        pytest.skip("Hanif converted instance missing")
    from src.data.instance_loader import load_instance

    inst = load_instance(hanif)
    assert inst.station_wait_mu is not None
    import numpy as np
    finite = inst.station_wait_mu[np.isfinite(inst.station_wait_mu)]
    assert float(np.max(np.abs(finite))) > 0
    cloned = instance_without_station_wait(inst)
    assert float(np.max(np.abs(cloned.station_wait_mu))) == 0.0
    assert float(np.max(np.abs(cloned.station_wait_expected))) == 0.0
    assert float(np.max(np.abs(finite))) > 0
    expected = inst.station_wait_expected[np.isfinite(inst.station_wait_expected)]
    assert abs(float(np.mean(expected)) - float(np.mean(finite))) > 1e-6


def test_compare_marks_incomplete(tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_hanif_results",
        Path(__file__).resolve().parents[1] / "scripts" / "compare_hanif_results.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    compare = mod.compare

    def write_csv(path: Path, names: list[str], k: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["instance", "used_k", "runtime_s", "waiting_time", "feasible"]
            )
            w.writeheader()
            for name in names:
                w.writerow({
                    "instance": name,
                    "used_k": k,
                    "runtime_s": 1.0,
                    "waiting_time": 0.0,
                    "feasible": "true",
                })

    frozen = tmp_path / "frozen"
    cand = tmp_path / "cand"
    names = ["C1_10_1", "R1_10_1"]
    write_csv(frozen / "original_tw" / "results.csv", names, 80)
    write_csv(frozen / "relaxed_tw" / "results.csv", names, 60)
    write_csv(cand / "original_tw" / "results.csv", names[:1], 90)
    write_csv(cand / "relaxed_tw" / "results.csv", names[:1], 70)
    report = compare(cand, frozen)
    assert report["complete"] is False
    assert "candidate_mean_k" not in report["variants"]["A"]
    assert "partial_statistics_do_not_cite" in report["variants"]["A"]
