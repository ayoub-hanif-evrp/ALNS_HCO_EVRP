# Candidate figure manifest (Hanif ALNS-HCO)

PNG directory: `outputs/figures_candidates_png/` (PNG only).
Numerical extracts: `outputs/processed/candidates/`.
Source of truth: `outputs/raw/master_results.csv`.
Generator: `scripts/generate_hanif_candidate_figures.py`.
Drawing code: `src/alns-hco/candidate_figures.py`.

No solver rerun. No Schneider/CARE-FI figures. No dataset-description plots.
Figure 2 is a methodological illustration; no saved Hanif routes exist.
Percentage changes are always $100(\mathrm{new}-\mathrm{baseline})/\mathrm{baseline}$; negative = reduction.

| Fig | File | Research question | Data | Metrics | Comparison | Recommendation |
|---|---|---|---|---|---|---|
| 1 | `fig01_two_phase_framework.png` | How is ALNS-HCO organised as a two-phase algorithm? | schematic | architecture | none | Main |
| 2 | `fig02_hco_decision.png` | What is distinctive about an HCO charging decision? | schematic | Pbar_s, Wbar_s, r_s, detour, e_s | illustrative | Main |
| 3 | `fig03_partial_recharging_effect.png` | How much of the charging improvement comes from partial recharging alone? | master_results.csv | D, T, N_ch, E_ch, T_wait, T_ch, C_ch | ALNS-PR vs ALNS-FR | Main |
| 4 | `fig04_charging_ablation.png` | At which algorithmic stage do charging gains appear? | master_results.csv | E_ch, T_ch, C_ch, T_wait | ALNS-FR = 100; FR to PR to PR+HCO to HCO | Main |
| 5 | `fig05_alns_hco_paired_effect.png` | What does Phase-II ALNS-HCO change relative to ALNS-PR? | master_results.csv | K, D, T, N_ch, E_ch, T_wait, T_ch, C_ch | ALNS-HCO vs ALNS-PR | Main |
| 6 | `fig06_win_tie_loss.png` | Are the Phase-II effects consistent across the 60 instances? | master_results.csv | K, D, T, N_ch, E_ch, T_wait, T_ch, C_ch | HCO better / tie / PR better | Main |
| 7 | `fig07_family_hco_effect.png` | Which Hanif families benefit from charging-aware search? | master_results.csv | K, D, N_ch, E_ch, T_wait, C_ch | ALNS-HCO vs ALNS-PR by family | Main or Supplement |
| 8 | `fig08_family_partial_charging.png` | Is the partial-recharging benefit uniform across families? | master_results.csv | D, T, N_ch, E_ch, T_ch, C_ch | ALNS-PR vs ALNS-FR by family | Supplement |
| 9 | `fig09_fleet_distance_tradeoff.png` | How do the methods sit on the fleet-distance frontier? | master_results.csv | K, D | GI, VNS, SA, ALNS-FR, ALNS-PR, ALNS-HCO | Supplement |
| 10 | `fig10_expense_waiting_tradeoff.png` | Does lower station waiting imply lower charging expense? | master_results.csv | C_ch, T_wait | ALNS-FR to PR to PR+HCO to HCO | Supplement |
| 11 | `fig11_instance_expense_waiting.png` | On which instances does HCO reduce both waiting and expense? | master_results.csv | dC_ch, dT_wait | ALNS-HCO vs ALNS-PR, instance level | Supplement |
| 12 | `fig12_energy_expense_mechanism.png` | Do expense changes track charged energy, or station selection? | master_results.csv | dE_ch, dC_ch | ALNS-HCO vs ALNS-PR, instance level | Supplement |
| 13 | `fig13_visits_waiting_mechanism.png` | Does waiting reduction come mainly from fewer charging visits? | master_results.csv | dN_ch, dT_wait | ALNS-HCO vs ALNS-PR, instance level | Supplement |
| 14 | `fig14_charging_time_per_energy.png` | Does HCO change the effective charging time per unit energy? | master_results.csv | T_ch / E_ch | family means, ALNS-PR vs ALNS-HCO | Supplement |
| 15 | `fig15_seed_stability.png` | How repeatable are the stochastic methods across the five seeds? | master_results.csv | CV of D and C_ch | VNS, SA, ALNS-FR, ALNS-PR, ALNS-HCO | Supplement |

## Regeneration

```
python scripts/generate_hanif_candidate_figures.py
```
