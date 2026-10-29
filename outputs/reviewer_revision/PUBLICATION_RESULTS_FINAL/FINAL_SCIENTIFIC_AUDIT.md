# Final scientific audit

Package produced by \scripts/final_reviewer_completion.py\ on branch eviewer-revision-2026\.

## Executed outcomes
- True fixed-route FR/PR policy experiment: n=240, both_feasible=42, counterfactual_full_infeasible=198
- Schneider source field fidelity: {'n_instances': 56, 'n_ok': 56, 'n_with_errors': 0, 'distance_convention': 'euclidean unrounded (float32 matrices); travel=d/v; energy=d*r', 'literature_often_uses': 'arc distances rounded (commonly 2 decimals) before summing; verifier rounds total cost to 3 decimals'}
- Schneider unserved diagnosis: {'n_runs': 72, 'n_feasible': 0, 'reused': True}
- Word integration: captions=4, images=4, fig_counts={'Figure 2': 1, 'Figure 3': 1, 'Figure 4': 1, 'Figure 5': 1}, annex=False, pdf=skipped:ModuleNotFoundError

## Validation
- pytest: 52 passed, 10 skipped
- verify_no_algo_diff: OK (no protected algorithmic files vs origin/main)

## Honest limitations retained
- Schneider BKS gaps not claimed (see SCHNEIDER_BENCHMARK_COMPARABILITY.md)
- Full-solver FR/PR is not a pure recharge-policy ablation; isolation uses the controlled same-route experiment
- HCO k=5 is a fixed design choice (sensitivity 2/5/8 identical on development subset)
- Aggregate PhaseII==PhaseI metrics do not prove route identity
- Word PDF visual verification unconfirmed (win32com unavailable)
