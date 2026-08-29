"""Stage-2 station refinement, operational candidate pool, and scientific gate."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.hanif_protocols import (
    PROTOCOL_EXTENDED,
    PROTOCOL_PROPOSED_EXTENDED,
    evaluation_protocol,
)
from src.evaluation.paper_objective import expected_wait_s
from src.evaluation.validation_gate import decision_payload, evaluate_gate
from src.search.station_candidates import balanced_union, geo_nearest_stations, operational_station_pool
from src.search.station_refine import acceptable_refinement, visit_phi_burden

HANIF = Path("dataset/Hanif/converted_hanif_original_tw/C1_10_1")
GEN = Path("src/visualization/final_paper_figures.py")
GEN_SCRIPT = Path("scripts/generate_final_paper.py")


def _write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _row(name, k, distance, wait, tchg, cost, *, feasible=True, served=1000):
    return {
        "instance": name,
        "feasible": "true" if feasible else "false",
        "used_k": k,
        "distance": distance,
        "station_waiting_time": wait,
        "charging_time": tchg,
        "expected_charging_cost": cost,
        "served": served,
        "n_customers": served,
        "served_ratio": 1.0,
    }


def test_proposed_protocol_marks_price_in_objective():
    assert PROTOCOL_EXTENDED["charging_price_in_objective"] is False
    assert PROTOCOL_PROPOSED_EXTENDED["charging_price_in_objective"] is True
    classical = evaluation_protocol({"paper_objective": {"mode": "classical_distance"}})
    proposed = evaluation_protocol({"paper_objective": {"mode": "expected_heterogeneous"}})
    assert classical["charging_price_in_objective"] is False
    assert proposed["charging_price_in_objective"] is True
    assert proposed["expected_wait_in_objective"] is True


def test_gate_rejects_half_percent_charge_time_blip(tmp_path: Path):
    names = [f"C1_10_{i}" for i in range(1, 4)] + [f"R1_10_{i}" for i in range(1, 4)]
    for tw in ("original_tw", "relaxed_tw"):
        c_rows = [_row(n, 80, 50000, 600, 17000, 9000) for n in names]
        # +0.97% wait, -0.59% charge time, +0.15% expense — the previous false PASS.
        p_rows = [_row(n, 80, 49960, 605.8, 16899.7, 9013.5) for n in names]
        _write_results(tmp_path / "classical" / tw / "results.csv", c_rows)
        _write_results(tmp_path / "proposed" / tw / "results.csv", p_rows)
    gate = evaluate_gate(tmp_path / "classical", tmp_path / "proposed")
    assert gate["pass"] is False
    assert gate["scientific_pass"] is False
    assert gate["pass"] == gate["scientific_pass"]
    decision = decision_payload(gate, out_dir=tmp_path)
    assert decision["scientific_pass"] is False
    assert "PARTIAL" in decision["decision"]


def test_gate_rejects_fleet_increase_on_any_instance(tmp_path: Path):
    names = ["C1_10_1", "C1_10_2"]
    for tw in ("original_tw", "relaxed_tw"):
        c_rows = [_row(n, 80, 1000, 100, 100, 100) for n in names]
        p_rows = [
            _row("C1_10_1", 80, 1000, 90, 90, 90),
            _row("C1_10_2", 81, 1000, 90, 90, 90),
        ]
        _write_results(tmp_path / "classical" / tw / "results.csv", c_rows)
        _write_results(tmp_path / "proposed" / tw / "results.csv", p_rows)
    gate = evaluate_gate(tmp_path / "classical", tmp_path / "proposed")
    assert gate["pass"] is False
    assert any("fleet increased" in r for r in gate["reasons"])


def test_gate_passes_clear_station_improvement_same_fleet(tmp_path: Path):
    names = ["C1_10_1", "R1_10_1"]
    for tw in ("original_tw", "relaxed_tw"):
        c_rows = [_row(n, 70, 10000, 1000, 20000, 12000) for n in names]
        p_rows = [_row(n, 70, 10050, 950, 19400, 11600) for n in names]
        _write_results(tmp_path / "classical" / tw / "results.csv", c_rows)
        _write_results(tmp_path / "proposed" / tw / "results.csv", p_rows)
    gate = evaluate_gate(tmp_path / "classical", tmp_path / "proposed")
    assert gate["pass"] is True
    assert gate["scientific_pass"] is True
    assert gate["pass"] == gate["scientific_pass"]


def test_balanced_union_keeps_later_criteria():
    wait = [1, 2, 3, 4]
    price = [1, 2, 5, 6]
    rate = [1, 7, 8]
    detour = [9, 10]
    combined = [1, 11]
    sequential = []
    seen = set()
    for seq in (wait, price, rate, detour, combined):
        for s in seq:
            if s not in seen:
                seen.add(s)
                sequential.append(s)
    sequential = sequential[:8]
    merged = balanced_union([wait, price, rate, detour, combined], 8, quota=2)
    assert 9 in merged or 10 in merged
    assert 11 in merged
    assert sequential[-1] not in {9, 10, 11} or 11 not in sequential[:8]


def test_visit_phi_burden_does_not_add_raw_wait_and_price():
    class _Inst:
        station_wait_expected = {0: 100.0, 1: 1.0}
        station_price_expected = {0: 1.0, 1: 100.0}
        station_charging_rate = {0: 1.0, 1: 1.0}

    from src.evaluation.paper_objective import ObjectiveReference

    ref = ObjectiveReference(distance=1.0, tstation=10.0, ccharge=10.0)
    high_wait = visit_phi_burden(_Inst(), 0, 1.0, ref)
    high_price = visit_phi_burden(_Inst(), 1, 1.0, ref)
    assert high_wait != expected_wait_s(_Inst(), 0) + 1.0
    assert abs(high_wait - high_price) < 1e-9


def test_acceptable_refinement_never_adds_ev():
    score = SimpleNamespace(phi=2.90)
    good = SimpleNamespace(feasible=True, unserved_customers=0, hard_violation_count=0, used_evs=80, score=score)
    assert acceptable_refinement(replay=good, k_star=80, current_phi=3.0) is True
    extra = SimpleNamespace(feasible=True, unserved_customers=0, hard_violation_count=0, used_evs=81, score=score)
    assert acceptable_refinement(replay=extra, k_star=80, current_phi=3.0) is False
    worse_phi = SimpleNamespace(feasible=True, unserved_customers=0, hard_violation_count=0, used_evs=80, score=SimpleNamespace(phi=3.01))
    assert acceptable_refinement(replay=worse_phi, k_star=80, current_phi=3.0) is False


def test_operational_pool_is_not_distance_only():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    from src.data.instance_loader import load_instance

    inst = load_instance(HANIF)
    cust = next(i for i, n in enumerate(inst.nodes) if n.is_customer)
    depot = int(inst.depot_index)
    geo = geo_nearest_stations(inst, cust, depot, k=5)
    pool = operational_station_pool(inst, cust, depot, k_each=4, k_total=12)
    assert pool
    extra = set(pool) - set(geo)
    wait_geo = min(expected_wait_s(inst, s) for s in geo)
    wait_pool = min(expected_wait_s(inst, s) for s in pool)
    assert extra or wait_pool < wait_geo - 1e-9
    assert wait_pool <= wait_geo + 1e-9


def test_figures_have_no_inset_and_three_routes():
    text = GEN.read_text(encoding="utf-8")
    assert "inset_axes" not in text
    assert "return chosen[:3]" in text
    assert "Charging visits versus expected station waiting" in text
    assert "Shares are summed over all Proposed instances" in text
    assert "SoC before charging" in text


def test_paper_folder_is_gated():
    text = GEN_SCRIPT.read_text(encoding="utf-8")
    assert "Refusing to write outputs/figures/" in text
    assert "--frozen-final" in text
