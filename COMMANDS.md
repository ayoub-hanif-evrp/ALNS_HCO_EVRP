# Commands

Run from the project root.

---

## 1. Quick test

```powershell
python -m compileall src
pytest -q
python main_train.py --phase0-only
```

---

## 2. Train — RC only (current experiment)

RC `_21` / 100 customers · 10 cycles · exact EVRPTW · no C/R · no mixed macro-updates.

```powershell
python main_train.py --config configs/train_RC21_10cycles.yaml --device cuda
```

Expected startup log:

```text
Training phase: RC only
Train instances: RC _21 only
Cycles: 10
Mixed macro updates: disabled
C/R instances: excluded
```

Holdout eval on reserved RC `_21` instances runs automatically after training.

Manual holdout re-eval:

```powershell
python -c "from src.evaluation.evaluate import evaluate_from_config_path; evaluate_from_config_path('configs/train_RC21_10cycles.yaml')"
```

Optional resume:

```powershell
python main_train.py --config configs/train_RC21_10cycles.yaml --device cuda --resume outputs/checkpoints/last.pt
```

---

## 3. Train — mixed C + R + RC

Shared model on all classes (`configs/train.yaml`). Deterministic C→R→RC macro-updates.

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

Optional resume:

```powershell
python main_train.py --config configs/train.yaml --device cuda --resume outputs/checkpoints/last.pt
```

---

## 4. Dashboard

Second terminal (during or after training):

```powershell
python main_dashboard.py
```

Open http://localhost:8080
