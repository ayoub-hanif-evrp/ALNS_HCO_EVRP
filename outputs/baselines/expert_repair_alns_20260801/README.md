# Frozen deterministic milestone: Expert+Repair+ALNS

**Immutable.** Do not overwrite.

This baseline is the successful **deterministic** solver
`Expert+Repair+ALNS` evaluated once on the paper full-9 holdouts.

It is **not** Transformer-guided TG-FPS. Neural checkpoints were not
loaded on this inference path (`neural_variants: 0`).

| Metric | Value |
|---|---:|
| Method | Expert+Repair+ALNS |
| Served | 0.998 |
| Feasibility | 0.889 (8/9) |
| Mean paper J | 62371.0 |

Source tree: `outputs/tgfps_full9/` (copied here).
See `manifest.json` for SHA256 of the holdout summary.
