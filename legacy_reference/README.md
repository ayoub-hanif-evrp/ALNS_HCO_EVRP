# Legacy ALNS-HCO reference

Frozen copy of the implementation and 120-instance results **before** the
paper-methodology revision (one HCO, Phase-I ALNS-PR, Phase-II freeze \(K=K^*\)).

| Path | Contents |
|---|---|
| `alns-hco-code/` | `src/alns-hco` as of that freeze |
| `alns-hco-outputs/` | `outputs/alns-hco` CSV/Markdown/JSON |
| `alns-hco.zip` | the two folders above |

Do not overwrite these files. New paper runs write to `outputs/hanif/`.
The live citeable snapshot `outputs/alns-hco/` is also left in place.

That legacy solver was allowed to **reduce** fleet in Phase II and started
Phase I from full-recharge ALNS plus compression. It is not the paper method.
