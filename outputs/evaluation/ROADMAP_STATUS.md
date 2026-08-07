# Schneider curriculum & paper packaging

## Type-2 (ready)

Anti-drift run: `outputs/runs/run_20260729_140606`

| Item | Result |
|---|---|
| Best train cycle | **4** — served 0.997 / feas 0.700 / J ≈ 33637 |
| Holdout (greedy) | served **1.000** / feas **3/3** on `c208_21`, `r211_21`, `rc208_21` |
| Best holdout J | `final_best` (`checkpoints/best.pt`) mean J ≈ **39744**, feas **3/3** |
| Demos | `outputs/demonstrations/schneider21_type2_v3_expert.pt` (24/24 feas) |

Do **not** claim full Schneider success from type-2-only numbers.

Artifacts:
- `outputs/runs/run_20260729_140606/evaluation/antidrift_cycle_summary.md`
- `outputs/runs/run_20260729_140606/evaluation/paper_table_type2.md`
- `outputs/evaluation/paper_holdout_tables.md`

## Type-1 / curriculum (wired, teacher gate pending)

Upgraded expert (earliest-due type-1 ranking, analytic critical lookahead, repair reopen):
- Smoke @ m=45: mean served **0.963**, feas **0.0** (gate ≥0.5 not met)
- Details: `outputs/expert_diagnostics/type1_smoke.json`

Curriculum entrypoint (type2 warmup → mix type1, full 9 holdouts, warm-start from type-2 best):

```bash
python main_train.py --config configs/train_curriculum.yaml --device cuda
```

Only launch after type-1 smoke feas clears ~0.5–0.7, then:

```bash
python _regen_demos.py   # regenerates curriculum demos when gate passes
```

Full-9 holdouts: `c101/c104/c208`, `r101/r104/r211`, `rc101/rc104/rc208`.

## Phase-1 polish (optional)

Already enabled in `configs/train.yaml` for any future type-2 polish run:
- `bc_regularization.loss_mode: reverse_kl`
- `use_multi_sample_rollout_baseline: true`, `multi_sample_K: 4`

Not required now: holdout feas is already 3/3.
