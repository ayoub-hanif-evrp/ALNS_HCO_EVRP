# Old vs new ALNS-HCO

The frozen solver in `legacy_reference/` (and `outputs/alns-hco/`) is **not** the paper method.

| | Legacy snapshot | Paper method |
|---|---|---|
| Phase I | Full-recharge ALNS + fleet compression | ALNS-PR (just-enough partial recharge) |
| Phase II fleet | May decrease \(K\) | Frozen \(K=K^*\) exactly |
| Oracle | `fast_hco` / `exact_hco` names | One function `hco` |
| Budget vs ALNS | 15 s ALNS vs ~33 s HCO | ALNS-PR 30 s vs ALNS-HCO 15+15 s |
| Seeds | 1 (`seed=0`) | 5 (`0..4`) |

Do not mix rows from `outputs/alns-hco/results.csv` into `outputs/hanif/raw/master_results.csv`.
Re-run `python scripts/run_all_paper_experiments.py` to fill the paper master file.
