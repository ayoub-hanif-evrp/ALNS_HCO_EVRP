# Reproducing the original published Hanif / ALNS-HCO results

This document freezes how to regenerate the **original** paper numbers without
changing algorithms, hyperparameters, or accepted solutions.

## Frozen scientific method

| Component | Location | Notes |
|---|---|---|
| Phase I (ALNS-PR) | `src/alns-hco/phase1.py`, `alns_pr.py` | Lex `(unserved, K, distance)`; just-enough partial recharge |
| ALNS-FR | `src/care-fi/benchmarks/alns.py` | Full recharge baseline |
| Fixed-route HCO | `src/alns-hco/hco.py` → `solution_from_hco` | Customer sequences frozen |
| Phase II | `src/alns-hco/phase2.py` | Fleet `K=K*`; distance gate `(1+ε)D^{PR}` |
| HCO-Detour / Heterogeneous | `src/alns-hco/hco.py` | Same `hco(..., mode=...)` |
| Budgets / seeds | `src/common/config.py` | 30 s total; 15+15 split; seeds `{0,1,2,3,4}` |
| Frozen config dump | `outputs/config/frozen_experiment_config.json` | Do not edit for original reproduction |

Method narrative: `src/alns-hco/METHOD.md`, `METHOD_ALNS_HCO.md`.

## Do not overwrite trusted artifacts

- Keep `outputs/tables/`, `outputs/statistics/`, `outputs/shared_phase1/`, and
  any dated baseline freezes under `outputs/baselines/` intact.
- New reviewer material belongs under `outputs/reviewer_revision/` only.
- Round-1 reporting fields are **post-hoc**; they must not feed ALNS/HCO decisions.

## Environment

```powershell
cd <repo-root>
python -m pip install -r requirements.txt
python -c "import src; print('ok')"
```

Requires converted Hanif folders:

- `dataset/Hanif/converted_hanif_original_tw/<INSTANCE>/`
- `dataset/Hanif/converted_hanif_relaxed_tw/<INSTANCE>/`

each with `evrp_instance.json`, `nodes.csv`, `stations.csv`, `matrices.npz`.

## Lightweight smoke (does not regenerate paper tables)

```powershell
python -m pytest tests/test_hco_final_fix.py tests/test_alns_hco.py tests/test_energy_accounting.py -q
```

Optional short solve (2 s + 2 s, not paper budget):

```powershell
python -c "from src.hanif.loader import load_hanif; from src.alns_hco.solver import solve_alns_hco; r=solve_alns_hco(load_hanif('C1_10_1','original'), mode='alns_hco', phase1_s=2.0, phase2_s=2.0, seed=0); print(r.metrics.feasible, r.metrics.used_k, r.runtime_s)"
```

## Full original experiment regeneration (expensive; not Round 1)

```powershell
python -m pytest tests/test_hco_final_fix.py -q
python scripts/select_phase2_epsilon.py
# Only if trusted baselines are not already preserved:
python scripts/preserve_trusted_baselines.py
python scripts/run_final_hco_branches.py --workers 8
python scripts/generate_final_paper_figures.py
```

Equal-budget claim: ALNS-PR 30 s vs ALNS-HCO 15 s Phase I + 15 s Phase II.

## Round-1 reporting extensions

New fields (`propulsion_energy`, `enroute_purchased_energy`, residual / depot
replenishment, ALNS iterations, HCO calls, execution times) are computed in
`src/common/energy_accounting.py` and attached in `clean_experiments.py` /
`station_log.py` **after** the solve. They do not alter operators, scores, or
acceptance rules.
