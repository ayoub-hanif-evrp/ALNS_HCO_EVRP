# Numerical pack for the writing agent

Cite only methods listed under `methods_present`. Never invent competitor means.

Archive: `C:/Users/ayoub/Desktop/Transformer_MARL_EVRP-main/outputs`
Methods present: ALNS, Classical, Constructor, Proposed, VNS-TS
Methods missing (do not cite): (none)

**Fleet comparison (mean K, 60/60 feasible each):**

| Method | Original TW | Relaxed TW |
|---|---:|---:|
| Constructor | 94.82 | 94.82 |
| ALNS-style | 93.58 | 91.53 |
| VNS-style | 94.73 | 94.68 |
| Classical / Proposed | **78.82** | **65.20** |

Proposed keeps Classical’s fleet and then cuts charging time/expense (see table3).
Caveat: Constructor Relaxed-TW metrics match Original TW because A→B warm start was accepted and intensification was off, so B did not re-descend the fleet. Do not describe constructor B as an independent relaxed-TW optimum.

**Do not write** that CARE-FI beat Keskin & Çatay or Schneider numerically. ALNS-style / VNS-style are matched internals, not the published algorithms.
Do not claim two-phase fleet-first as novelty (Wang et al. 2026).

See `architecture.md` for the method and `outputs/COMPARISON_FOR_PAPER.md` for the literature/baseline comparison.
- `outputs/tables/table1_overall.csv`
- `outputs/tables/table2_family.csv`
- `outputs/tables/table3_objective_comparison.csv`
- `outputs/tables/table4_method_comparison.csv`
- `outputs/tables/table5_energy_vs_expense.csv`

- Original TW: ΔK=0.000, distance -0.61%, wait -3.77%, charge-time -36.40%, expense -42.38%, energy -37.85%, customer-wait +5.15% (wait improved on 46/60, expense 60/60).
- Relaxed TW: ΔK=0.000, distance -0.96%, wait -8.23%, charge-time -30.06%, expense -39.69%, energy -33.04%, customer-wait +7.31% (wait improved on 54/60, expense 60/60).
