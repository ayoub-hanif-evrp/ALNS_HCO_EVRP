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

## 2. Full training

One shared model on Schneider **100-customer** (`_21`) instances from C, R, and RC.

Deterministic order each cycle (filename-sorted, never shuffled):

```text
C1 -> R1 -> RC1 -> C2 -> R2 -> RC2 -> ...
```

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

Optional resume after an interrupt:

```powershell
python main_train.py --config configs/train.yaml --device cuda --resume outputs/checkpoints/last.pt
```

---

## 3. Dashboard

Run in a **second terminal** while training (or after):

```powershell
python main_dashboard.py
```

Open http://localhost:8080
