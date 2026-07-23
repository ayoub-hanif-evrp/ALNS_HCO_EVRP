# Commands

Run from the project root.

## Full pipeline (start from scratch)

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

Tuned for this PC (24 CPU + RTX A5000):
- Stage 1 demos: **13 parallel CPU workers** (GPU stays ~0% — exact masks are CPU)
- BC cache: same parallel CPU workers
- BC / RL: **CUDA + AMP + TF32** (GPU usage rises here)

## Dashboard

```powershell
python main_dashboard.py
```
