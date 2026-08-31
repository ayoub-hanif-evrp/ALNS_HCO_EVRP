"""Determinism, feasibility, and record-shape tests for Hanif CARE-FI."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.hanif.instance import load_instance
from src.hanif.output import validate_record
from src.hanif.construction import construct_feasible, customer_order
from src.hanif.replay import replay_route, replay_solution, sequences_match
from src.hanif.solver import result_record, solve_carefi, solve_stage2

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101_21"
HANIF_C1 = ROOT / "dataset" / "Hanif" / "converted_hanif_original_tw" / "C1_10_1"


@pytest.fixture(scope="module")
def tiny():
    if not TINY.is_dir():
        pytest.skip("c101C5 missing")
    return load_instance(TINY)


def test_customer_order_is_deterministic(tiny):
    a = customer_order(tiny)
    b = customer_order(tiny)
    assert a == b
    dues = [tiny.nodes[i].due_time for i in a]
    assert dues == sorted(dues)


def test_replay_empty_and_single_customer(tiny):
    depot = tiny.depot_index
    empty = replay_route(tiny, [depot, depot])
    assert empty.feasible
    custs = [i for i, n in enumerate(tiny.nodes) if n.is_customer]
    plan = replay_route(tiny, [depot, custs[0], depot], default_portion=1.0)
    assert plan.customers == [custs[0]] or not plan.feasible


def test_construct_and_solve_are_deterministic(tiny):
    r1 = construct_feasible(tiny, nearby_routes=6, insert_positions=3, nearby_stations=3)
    r2 = construct_feasible(tiny, nearby_routes=6, insert_positions=3, nearby_stations=3)
    assert [(tuple(n), tuple(l)) for n, l in r1] == [(tuple(n), tuple(l)) for n, l in r2]
    s1a, s2a = solve_carefi(tiny, budget_s=20.0)
    s1b, s2b = solve_carefi(tiny, budget_s=20.0)
    assert s1a.metrics.used_k == s1b.metrics.used_k
    assert abs(s1a.metrics.distance - s1b.metrics.distance) < 1e-6
    if s2a and s2b:
        assert sequences_match(tiny, s1a.routes, s2a.routes)
        assert s2a.metrics.used_k == s1a.metrics.used_k


def test_stage2_does_not_worsen_phi_or_k(tiny):
    s1, s2 = solve_carefi(tiny, budget_s=20.0)
    assert s1.metrics.feasible
    assert s2 is not None
    assert s2.metrics.used_k == s1.metrics.used_k
    assert sequences_match(tiny, s1.routes, s2.routes)
    assert s2.validation.get("phi_nonworse") is True


def test_result_record_has_required_fields(tiny):
    s1, s2 = solve_carefi(tiny, budget_s=12.0)
    row = result_record(tiny, s1, tw="original", configuration={"budget_s": 5})
    validate_record(row, stage2=False)
    if s2:
        row2 = result_record(tiny, s2, tw="original", configuration={"budget_s": 5})
        validate_record(row2, stage2=True)


def test_exact_method_has_no_two_opt_or_regret():
    import src.hanif.fleet_reduction as fr
    import src.hanif.construction as cons

    assert not hasattr(fr, "_two_opt")
    src = Path(fr.__file__).read_text(encoding="utf-8")
    assert "regret_insertion" not in src
    assert "random" not in src
    assert Path(cons.__file__).read_text(encoding="utf-8").count("delta_d") >= 1


def test_stage2_candidates_are_the_five_stations(tiny):
    from src.hanif.charging_refinement import candidate_stations
    from src.hanif.construction import construct_feasible

    routes = construct_feasible(tiny, nearby_routes=4, insert_positions=2, nearby_stations=2)
    found = False
    for nodes, _ in routes:
        for pos, n in enumerate(nodes):
            if tiny.nodes[n].is_charging_station:
                cand = candidate_stations(tiny, nodes, pos)
                assert 1 <= len(cand) <= 5
                found = True
                break
        if found:
            break


def test_current_solver_still_importable(tiny):
    from src.hanif.current.solver import solve_carefi as solve_v1

    s1, _ = solve_v1(tiny, budget_s=8.0, stage2=False)
    assert s1.metrics.used_k >= 1


@pytest.mark.skipif(not HANIF_C1.is_dir(), reason="Hanif C1_10_1 not converted")
def test_hanif_replay_accepts_frozen_classical():
    from src.hanif.solver import load_archived_classical

    inst = load_instance(HANIF_C1)
    routes = load_archived_classical("C1_10_1", "original")
    if routes is None:
        pytest.skip("frozen classical C1_10_1 missing")
    check = replay_solution(inst, routes)
    assert check.feasible
    assert check.used_k >= 1
    s2 = solve_stage2(inst, routes, max_passes=1)
    assert s2.metrics.feasible
    assert s2.metrics.used_k == check.used_k
    assert sequences_match(inst, routes, s2.routes)
