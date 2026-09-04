# Figure manifest (Hanif manuscript)

Source of truth: `outputs/raw/master_results.csv`
Generator: `scripts/generate_hanif_figures.py`
Drawing code: `src/alns-hco/figures.py`

Each figure answers one question. Detailed numbers remain in Tables 2–5.
All manuscript figures are **PNG only** (no PDF/SVG sidecars).
Schneider, CARE-FI, runtime, and dataset-description plots are not produced.
The Hanif dataset paper already describes station-price/waiting distributions
and the Original/Relaxed TW structure; those plots are not repeated here.

## Main manuscript

| Fig | File | Question | Source |
|---|---|---|---|
| 1 | author file `figures/fig01_alns_hco_framework.png` | How is ALNS-HCO organised? | **Author drawing. Not generated. Do not use the candidate framework PNG.** |
| 2 | `figures/fig02_hco_decision.png` | How does HCO choose a station and a partial charge? | Conceptual schematic (not experimental) |
| 3 | `figures/fig03_partial_recharging_effect.png` | How much of the charging improvement comes from partial recharging alone? | Paired ALNS-FR → ALNS-PR forest + 95% CI |
| 4 | `figures/fig04_paired_effects.png` | How does full ALNS-HCO compare with a 30 s ALNS-PR? | Paired instance means + bootstrap 95% CI. The $K$ gap is the 15 s vs 30 s Phase-I budget, not HCO changing fleet size |
| 5 | `figures/fig05_family_charging.png` | Which families gain or lose on charging metrics? | Same pairing, families C1–RC2; columns $N_{ch}$, $T_{wait}$, $C_{ch}$ only |

Figures 2–5 are PNG only. Numeric extracts: `processed/<stem>.csv`.

## Supplement

| Fig | File | Question | Note |
|---|---|---|---|
| S1 | `figures/supplement/figS01_charging_ablation_cross_run.png` | Cross-run FR / PR / PR+HCO / HCO index profile | **Not** a controlled sequential ablation; prefer Table 4 Panel B for within-run HCO |

Candidate mechanism / WTL / stability plots remain in `figures_candidates_png/` for optional supplement selection.

## Not used in the manuscript

Dashboard figures and the superseded main four-stage ablation live in `figures/archive/`.
Runtime is uninformative at a matched 30 s budget.

No convergence plots (no iteration traces). No experimental route/SoC figure
(no saved route dumps). Figure 2 is labelled as an illustration for that reason.

## Regeneration

```
python scripts/generate_hanif_figures.py
```

This command never creates or overwrites Figure 1.
