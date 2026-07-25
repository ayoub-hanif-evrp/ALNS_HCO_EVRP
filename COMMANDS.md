# Commands

Run from the **project root**.

Study: **EVRP-TW-D** · Cus_100 · **100 easiest RC wide** train · **10 test** (5 wide + 5 narrow) · **1000** RL updates.

---

## Train

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

---

## Test

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage eval
```

---

## Dashboard

```powershell
python main_dashboard.py
```

Open **http://127.0.0.1:8080**
