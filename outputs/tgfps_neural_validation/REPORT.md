# Neural integration validation report

## Clarification on prior 8/9 result

The holdout result **8/9 feas, served 0.998, J≈62371** was achieved by
**`Expert+Repair+ALNS`** (deterministic). It was **not** Transformer-guided TG-FPS.

Frozen immutable copy: `outputs/baselines/expert_repair_alns_20260801/`.

## Question

> Does the Transformer improve the validated deterministic solver through
> better solution quality or lower search cost?

## Answer

**NO.** Keep **`Expert+Repair+ALNS`** as the deployed solver.

On all **6** reserved validation instances
(`c103_21, c201_21, r103_21, r201_21, rc103_21, rc201_21`):

| Method | Feas | Served | Mean J | Mean runtime |
|---|---:|---:|---:|---:|
| Expert+Repair+ALNS | 0.833 | 0.990 | 65,949 | 133s |
| Neural-Guided Expert+Repair | 0.500 | 0.983 | 105,004 | 513s |
| Neural-Guided Expert+Repair+ALNS | 0.500 | 0.983 | 104,969 | 537s |

Both neural methods **failed** the no-regression gate (`feasibility_regression`).
Full-9 was **not** re-run for neural (method frozen on validation = deterministic fallback).

## Proof the Transformer participated

Checkpoint: `outputs/baselines/full9_20260731/checkpoints/bc_best_policy.pt`  
SHA256: `c60f7380542870968be74bd86f6bb47ea2195e3fa6459e5756a26ded544785ae`

Telemetry (`raw/inference_telemetry.json`): all 12 neural-method rows have
`neural_participation: true`. Example (`c103_21`, Neural-Guided Expert+Repair):

- neural_decisions: 131
- score_candidate_calls: 93 (full Transformer `score_candidates`)
- accepted_neural_guided_moves: 131
- neural_expert_disagreements: 48
- expert_fallbacks: 0
- candidates_generated: 1961

Missing-checkpoint paths raise `NeuralCheckpointError` (no silent relabel).

## Artifacts

- Validation gate: `outputs/tgfps_neural_validation/hybrid_best_validation.json`
- Per-instance metrics: `outputs/tgfps_neural_validation/raw/method_comparison.csv`
- Telemetry: `outputs/tgfps_neural_validation/raw/inference_telemetry.json`
- Hash verify: `python scripts/verify_tgfps_hashes.py`
