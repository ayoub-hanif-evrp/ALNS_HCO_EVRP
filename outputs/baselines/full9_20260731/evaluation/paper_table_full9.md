# Full Schneider 9-holdout table (greedy decode, best ckpt)

| Method | Served | Feas | Mean J |
|---|---:|---:|---:|
| Expert | 0.984 | 0.667 | 96950.5 |
| BC | 0.920 | 0.667 | 161077.6 |
| RL-best* | 0.903 | 0.667 | 177774.5 |

Holdouts: c101_21, c104_21, c208_21, r101_21, r104_21, r211_21, rc101_21, rc104_21, rc208_21. * = checkpoint selected on validation/test protocol. Use hierarchical **best** checkpoint only; shaped reward is not paper J.
