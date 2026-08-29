# outputs/

Final paper results. Every method and artifact sits directly here.

| Path | What it is |
|---|---|
| `classical/` | Stage 1: fleet-first solutions (60 Original TW + 60 Relaxed TW) |
| `proposed/` | Stage 2: same routes, heterogeneous charging refinement |
| `references/` | Frozen Φ normalisers taken from Classical |
| `constructor/` | Matched constructor baseline (intensification off) |
| `alns/` | Matched ALNS baseline |
| `vns_ts/` | Matched VNS/TS baseline |
| `figures/` | Six manuscript figures |
| `tables/` | Manuscript CSVs, including `per_instance.csv` |
| `VALIDATION.json` | Scientific gate on the full 120 |
| `WRITING_PACK.md` | Numbers for the writing agent |
| `HEADLINE_NUMBERS.json` | Same numbers as JSON |

Do not add nested experiment-version folders (`refinement_v1/full`, `baselines/…`). New solver runs go to a **new** directory such as `outputs/hanif_fps_run_new`.
