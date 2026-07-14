# Commands

Run from the project root, **in order**. Stop if a step fails.

Workflow:

```text
single RC debugging  →  full RC baseline  →  balanced all-class generalization
```

There are exactly **three training phases**. Evaluation is not a training phase.

---

## 0. Sanity checks

This is **not** a training phase.

```powershell
python -m compileall src
pytest -q
python main_train.py --config configs/train_phase1_single_RC21.yaml --phase0-only
```

---

## 1. Phase 1 — Single RC instance

Purpose: verify learning, masks, SMDP returns, critic, and training speed on one RC `_21` instance (~500 updates).

```powershell
python main_train.py --config configs/train_phase1_single_RC21.yaml --device cuda
```

Phase 1 passes only if:

* served ratio improves
* objective improves
* critic value variance is nonzero
* explained variance begins improving
* no invalid actions
* no objective reconstruction mismatch
* no max-step loops

Optional live monitor (second terminal):

```powershell
python main_dashboard.py
```

---

## 2. Phase 2 — Full RC training

Train on **all** available RC `_21` training instances (holdouts reserved). One full RC baseline run (100 cycles).

```powershell
python main_train.py --config configs/train_phase2_full_RC21.yaml --device cuda
```

---

## 3. Phase 3 — All classes

After RC learning is stable, train jointly on C + R + RC with balanced class sampling:

```text
P(C) = P(R) = P(RC) = 1/3
```

```powershell
python main_train.py --config configs/train_phase3_all_classes.yaml --device cuda
```

Separate unseen holdouts are kept for C, R, and RC.

---

## 4. Evaluation

Evaluation is **not** a training phase.

### RC holdout after Phase 2

```powershell
python main_evaluate.py --config configs/train_phase2_full_RC21.yaml --checkpoint outputs/checkpoints/last.pt --device cuda --class RC
```

### Per-class holdouts after Phase 3

```powershell
python main_evaluate.py --config configs/train_phase3_all_classes.yaml --checkpoint outputs/checkpoints/last.pt --device cuda --class C
python main_evaluate.py --config configs/train_phase3_all_classes.yaml --checkpoint outputs/checkpoints/last.pt --device cuda --class R
python main_evaluate.py --config configs/train_phase3_all_classes.yaml --checkpoint outputs/checkpoints/last.pt --device cuda --class RC
```

### Combined class-balanced macro-average (after Phase 3)

```powershell
python main_evaluate.py --config configs/train_phase3_all_classes.yaml --checkpoint outputs/checkpoints/last.pt --device cuda --macro-average
```

Writes `outputs/results/class_macro_average.json` with per-class means and the macro-average.

---

## Optional — resume a phase

Do **not** mix these into the main sequence. Use only when continuing an interrupted run.

A true resume restores actor, critic, optimizers, normalizers, update/cycle counters, and RNG states when available.

```powershell
python main_train.py --config configs/train_phase1_single_RC21.yaml --device cuda --resume outputs/checkpoints/last.pt
python main_train.py --config configs/train_phase2_full_RC21.yaml --device cuda --resume outputs/checkpoints/last.pt
python main_train.py --config configs/train_phase3_all_classes.yaml --device cuda --resume outputs/checkpoints/last.pt
```
