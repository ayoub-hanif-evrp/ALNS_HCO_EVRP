# Figure manifest (Hanif manuscript)

Source of truth: `outputs/raw/master_results.csv`
Generator: `scripts/generate_hanif_figures.py`
Drawing code: `src/alns-hco/figures.py`

Each figure answers one question. Detailed numbers remain in Tables 2–5.
Schneider, CARE-FI, runtime, and dataset-description plots are not produced.
The Hanif dataset paper already describes station-price/waiting distributions
and the Original/Relaxed TW structure; those plots are not repeated here.

## Main manuscript

| Fig | File | Question | Source |
|---|---|---|---|
| 1 | author file `figures/fig01_alns_hco_framework` | How is ALNS-HCO organised? | **Author drawing. Not generated.** |
| 2 | `figures/fig02_hco_decision` | How does HCO choose a station and a partial charge? | Conceptual schematic (not experimental) |
| 3 | `figures/fig03_charging_ablation` | Where do charging gains come from? | ALNS-FR, ALNS-PR, ALNS-PR+HCO, ALNS-HCO |
| 4 | `figures/fig04_paired_effects` | What does Phase-II ALNS-HCO change vs ALNS-PR? | Paired instance means + bootstrap 95% CI |
| 5 | `figures/fig05_family_charging` | Which families gain or lose on charging metrics? | Same pairing, families C1–RC2; columns $N_{ch}$, $T_{wait}$, $C_{ch}$ only |

Figures 2–5 are saved as PDF, SVG, and PNG. Numeric extracts: `processed/<stem>.csv`.

## Not used in the manuscript

Dashboard figures from the previous generator are in `figures/archive/`:
paired boxplots, ten-panel charging bars, charging-expense scatter, seven-column
heatmap, and runtime distributions. Runtime is uninformative at a matched 30 s budget.

No convergence plots (no iteration traces). No experimental route/SoC figure
(no saved route dumps). Figure 2 is labelled as an illustration for that reason.

## Regeneration

```
python scripts/generate_hanif_figures.py
```

This command never overwrites Figure 1.
