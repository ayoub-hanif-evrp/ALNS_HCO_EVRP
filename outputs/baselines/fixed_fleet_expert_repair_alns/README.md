# FixedFleet-Expert+Repair+ALNS (immutable)

**Fleet minimization was NOT performed.**

This baseline evaluates `Expert+Repair+ALNS` at a **fixed fleet upper bound**
`m = 40`. Reported `used_evs` may equal the cap and must **not** be interpreted
as a minimized fleet size.

| Metric | Value |
|---|---:|
| Method | FixedFleet-Expert+Repair+ALNS |
| Served | 0.998 |
| Feasibility | 8/9 |
| Mean paper J | 62371 |
| Fleet protocol | FIXED_UPPER_BOUND (m=40) |

Do not overwrite. Successor methods that minimize fleet must be stored under a
different baseline id (e.g. `Deterministic-FPS`).
