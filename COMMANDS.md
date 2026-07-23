# Commands

Run from the project root.

## Full pipeline (Hanif-only pilot)

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

This single command runs:

1. Hanif original-TW conversion check (`variant_A` → `converted_hanif_original_tw`)
2. Dataset compatibility report + deterministic C/R/RC train/val/test splits (B3 pilot)
3. Internal sanity checks
4. Expert demos → decision-level cache (`hanif_expert_train.pt`)
5. Behavior cloning (full epochs, no early stopping)
6. Conservative RL fine-tuning from BC
7. Multi-checkpoint evaluation + `final_report.json` / `.md`

**Note:** Each Hanif instance has ~1000 customers + 333 stations. Exact-mask expert/RL on the pilot subset is still heavy. Do not mix Schneider.

## Dashboard

```powershell
python main_dashboard.py
```

Loads the latest `outputs/runs/run_*` (dataset report, splits, expert/BC/RL, checkpoint comparison). Does not train.
