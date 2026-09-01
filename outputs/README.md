# outputs/

Citeable archives only. Do not add smoke folders, versioned experiments, or nested run trees.

| Path | What it is |
|---|---|
| `care-fi/` | Citeable CARE-FI Hanif results |
| `alns-hco/` | Frozen pre-paper ALNS-HCO snapshot (do not overwrite) |
| `hanif/` | ALNS-HCO paper experiments (Hanif 1000-customer set) |
| `schneider/` | Schneider EVRPTW results (kept separate from Hanif) |

New solver runs write under `outputs/_tmp/` (gitignored). Delete that folder when you are done looking at the numbers.
