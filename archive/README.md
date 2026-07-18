# Archive

Files moved here are **not** part of the normal two-command workflow:

```powershell
python main_train.py --config configs/train.yaml --device cuda
python main_dashboard.py
```

They are kept for reference / optional developer use.

## Layout

| Folder | Contents |
|--------|----------|
| `configs/` | Obsolete YAML configs replaced by `configs/train.yaml` |
| `entrypoints/` | `main_evaluate.py`, `main_maps_dashboard.py` |
| `src_training/` | Standalone `warmstart_comparison.py` CLI (pipeline writes comparison itself) |
| `scripts/` | GPU check, benchmarks, diagnostics |
| `docs/` | Historical correctness/perf report |
| `misc/` | `context.txt`, `src.zip` |

To run an archived entrypoint:

```powershell
python archive/entrypoints/main_evaluate.py --config configs/train.yaml --checkpoint <path>
python archive/scripts/check_gpu.py
```
