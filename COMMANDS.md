# Commands

Run from the project root.

## 1. Train

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

This single command runs the full pipeline:

1. internal sanity checks
2. expert evaluation
3. expert demonstration generation
4. behavior cloning (fixed epoch count, no early stopping)
5. BC policy evaluation
6. RL fine-tuning from the BC checkpoint
7. final holdout evaluation
8. save logs, checkpoints, reports, and dashboard artifacts

Artifacts are written under:

```text
outputs/runs/run_<timestamp>/
```

## 2. Dashboard

```powershell
python main_dashboard.py
```

Automatically opens the latest run under `outputs/runs/` at http://localhost:8080.

---

## Optional developer checks

```powershell
python -m compileall src
pytest -q
```

Archived optional tools live under `archive/` (see `archive/README.md`).
