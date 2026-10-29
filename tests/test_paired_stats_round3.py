"""Regression tests for Round-3 pairing, completion, and runtime reporting."""
from __future__ import annotations

from src.common.paired_stats import paired_instance_effects, protocol_completion, summarize_paired
from src.alns_hco.job_store import job_id_fr_pr


def _row(**kw):
    base = {
        "instance": "C1_10_1",
        "tw": "original",
        "seed": 0,
        "budget_s": 30.0,
        "method": "ALNS-FR",
        "feasible": True,
        "error": None,
        "distance": 100.0,
        "k": 10,
    }
    base.update(kw)
    return base


def test_pairing_requires_matched_seed_and_budget():
    rows = [
        _row(method="ALNS-FR", seed=0, budget_s=30, distance=100),
        _row(method="ALNS-PR", seed=0, budget_s=30, distance=90),
        _row(method="ALNS-FR", seed=1, budget_s=30, distance=110),
        # PR seed1 missing → seed1 excluded from average
        _row(method="ALNS-FR", seed=0, budget_s=120, distance=80),
        _row(method="ALNS-PR", seed=0, budget_s=120, distance=70),
        # Wrong budget must not pair with 30s
        _row(method="ALNS-PR", seed=1, budget_s=300, distance=50),
    ]
    eff30 = paired_instance_effects(
        rows, method_a="ALNS-FR", method_b="ALNS-PR", metric="distance",
        tw="original", budget_s=30.0,
    )
    assert len(eff30) == 1
    assert eff30[0]["n_matched_seeds"] == 1
    assert abs(eff30[0]["a_mean"] - 100.0) < 1e-12
    assert abs(eff30[0]["diff_b_minus_a"] - (-10.0)) < 1e-12

    eff120 = paired_instance_effects(
        rows, method_a="ALNS-FR", method_b="ALNS-PR", metric="distance",
        tw="original", budget_s=120.0,
    )
    assert len(eff120) == 1
    assert abs(eff120[0]["diff_b_minus_a"] - (-10.0)) < 1e-12


def test_seed_average_before_instance_ci():
    rows = [
        _row(method="ALNS-FR", seed=0, distance=100),
        _row(method="ALNS-PR", seed=0, distance=80),
        _row(method="ALNS-FR", seed=1, distance=120),
        _row(method="ALNS-PR", seed=1, distance=100),
    ]
    eff = paired_instance_effects(
        rows, method_a="ALNS-FR", method_b="ALNS-PR", metric="distance",
        tw="original", budget_s=30.0,
    )
    assert eff[0]["n_matched_seeds"] == 2
    assert abs(eff[0]["a_mean"] - 110.0) < 1e-12
    assert abs(eff[0]["b_mean"] - 90.0) < 1e-12
    summ = summarize_paired(eff)
    assert summ["n_instances"] == 1
    assert abs(summ["mean_diff"] - (-20.0)) < 1e-12


def test_protocol_completion_distinguishes_subset():
    planned = [
        {"method": "ALNS-FR", "instance_name": "C1_10_1", "tw": "original", "seed": 0, "budget_s": 30},
        {"method": "ALNS-PR", "instance_name": "C1_10_1", "tw": "original", "seed": 0, "budget_s": 30},
        {"method": "ALNS-FR", "instance_name": "C1_10_1", "tw": "original", "seed": 0, "budget_s": 120},
    ]
    done = {job_id_fr_pr(planned[0]), job_id_fr_pr(planned[1])}
    st = protocol_completion(done, planned, job_id_fn=job_id_fr_pr)
    assert st["n_planned"] == 3
    assert st["n_completed"] == 2
    assert st["n_missing"] == 1
    assert st["complete_protocol"] is False
    assert len(st["missing_job_ids"]) == 1


def test_tw_mismatch_not_paired():
    rows = [
        _row(method="ALNS-FR", tw="original", distance=100),
        _row(method="ALNS-PR", tw="relaxed", distance=90),
    ]
    eff = paired_instance_effects(
        rows, method_a="ALNS-FR", method_b="ALNS-PR", metric="distance",
        tw="original", budget_s=30.0,
    )
    assert eff == []
