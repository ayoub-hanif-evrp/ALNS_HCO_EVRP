# Manuscript revision guide — ready-to-insert text

**Repository note:** No `.tex` / `.docx` manuscript file is present in this workspace. Text below is organized by **section purpose** for paste into the authors’ working draft. Do **not** invent page or line numbers. All numbers are from verified Round 3–5 CSVs under `outputs/reviewer_revision/`.

**Figure attribution (correct):**  
- Figure 4 / Figure 5 = **PhaseII-NoHCO → PhaseII-Heterogeneous** (`outputs/figures/final_figure_manifest.md`).  
- Do **not** attribute Figure 5 to FR→PR or PhaseI→PhaseII.

---

## A. Experimental setup (Methods / Computational experiments)

**Insert under:** Experimental design / Computational environment.

> Experiments were executed on Windows 11 (AMD64; Intel processor Family 6 Model 143) with Python 3.13 for the frozen Hanif publication runs and Python 3.12 for the reviewer-revision verification tooling. The primary Hanif study uses the 60-instance, 1000-customer EVRPTW benchmark of Hanif and Wallace with two **related** time-window variants — **Original TW** and **Relaxed TW**. These variants share customer locations and demands but differ in time-window tightness; they are **not** independent stochastic replications of the same base instance.
>
> Stochastic methods use integer seeds. For the full Hanif main comparison we report seeds \(\{0,1,2\}\) (2520 completed method×instance×TW×seed records across seven reported methods). The multi-budget FR/PR convergence protocol uses a balanced 12-instance subset (two instances per family), both TW variants, methods ALNS-FR and ALNS-PR, wall-clock budgets \(\{30,120,300\}\) seconds, and seeds \(\{0,1\}\) only — yielding **288** primary runs (24 observations per method×TW×budget cell; exactly two matched seeds per instance in paired comparisons). Seed-2 cells from earlier runs are archived separately and are **excluded** from primary convergence inference.
>
> Equal search budget means equal ALNS wall-clock time limits. Phase-I/Phase-II ALNS-HCO uses a 15 s + 15 s split of the common 30 s budget. Reported `solver_runtime_s` is solver wall-clock; post-hoc metric extraction is excluded from the search budget. Matched comparisons are formed at the (instance, TW, seed[, budget]) level; seeds are averaged within instance before instance-level bootstrap confidence intervals. Original and Relaxed TW are analysed **separately** and are never pooled as if they were independent base instances.
>
> The lexicographic routing objective prioritises fleet size \(K\) first and total distance second. Distance differences are reported unconditionally for transparency and, where sample size allows, discussed conditionally on fleet-size ties.

**Hardware source:** `outputs/final_experiment_report.md` / `outputs/config/frozen_experiment_config.json`.  
**Canonical CSVs:** `hanif_main_seeds012/`, `fr_pr_convergence_primary288/`.

---

## B. Original vs Relaxed TW (clarify related variants)

**Insert under:** Benchmark description.

> Following Hanif and Wallace, we evaluate each geographic instance under two time-window settings. **Original TW** uses the published windows; **Relaxed TW** widens windows according to the benchmark’s relaxed regime. Pairwise statistics never treat (Original, Relaxed) as two independent draws of one instance: each TW defines its own matched panel of 60 instances (Hanif main) or 12 instances (convergence subset).

**Tables:** `benchmark_characteristics/table_hanif_instances.csv`.

---

## C. FR/PR longer-budget convergence findings

**Insert under:** Results — Full vs partial recharging / Computational budget.

> Under equal wall-clock budgets on the 12-instance convergence panel (seeds 0–1), partial recharging (ALNS-PR) consistently purchases less en-route energy than full recharging (ALNS-FR); paired bootstrap CIs for en-route energy exclude zero at 30, 120, and 300 seconds on both TW panels (`fr_pr_convergence_primary288/paired_summaries.csv`).
>
> Fleet size remains the primary lexicographic criterion. On Relaxed TW, FR more often obtains smaller fleets (e.g. at 300 s, mean \(\Delta K = K_{\mathrm{PR}}-K_{\mathrm{FR}} = +1.08\), 95% CI \([0.58, 1.63]\)). Apparent distance advantages for PR are therefore **not** interpreted as unconditional superiority when fleets differ.
>
> We further summarise the **change** in the instance-level FR–PR gap from 30 s to 300 s. On Original TW, the mean change in the fleet-size gap is \(+1.17\) vehicles (PR−FR), 95% CI \([0.08, 2.33]\): longer budgets do not close PR’s relative fleet-size disadvantage on this panel. Distance gap changes have wide CIs that include zero on both TW panels (`final_stats/table_gap_change_30_to_300_summary.csv`). Instances with fleet-size ties at both budgets are few (2 Original, 1 Relaxed); conditional distance statements for those cases are descriptive only.
>
> ALNS-FR and ALNS-PR are distinct search implementations. Under the same wall-clock limits, FR completes substantially more ALNS iterations than PR (means ≈ 44 398 vs 12 156 at 300 s on the primary panel). Iteration counts are reported so that equal wall-clock time is not mistaken for equal neighbourhood throughput.

**Figures:** `fig_fr_pr_convergence*.png` in `fr_pr_convergence_primary288/`.

---

## D. Energy balance, depot replenishment, total electricity

**Insert under:** Energy accounting / Performance metrics.

> Let \(E_{\mathrm{enroute}}\) be observed energy purchased at public stations and \(E_{\mathrm{return}}\) the residual battery upon depot return. Under full departure batteries, no charging losses, and the verified identity
> \[
> E_{\mathrm{drive}} = K\cdot B_{\max} + E_{\mathrm{enroute}} - E_{\mathrm{return}},\qquad
> E_{\mathrm{depot}} = E_{\mathrm{drive}} - E_{\mathrm{enroute}},
> \]
> depot replenishment \(E_{\mathrm{depot}}\) is **reconstructed** when route-level SOC traces are unavailable and is labelled as such in the data release. Observed public charging expense uses station prices applied to \(E_{\mathrm{enroute}}\) only; it is **not** equal to total electricity including depot energy, nor to hypothetical tariff scenarios applied post hoc.
>
> Linear constant-rate partial charging reduces purchased en-route quantity relative to forced full recharges; it does **not**, by itself, create benefits from nonlinear state-of-charge-dependent charging speeds, which are outside the present model.

**Coverage:** all 2520 Hanif main rows carry residual and depot fields after labelled reconstruction (`table_energy_coverage.csv`).

---

## E. Absolute ALNS-HCO results and limited Phase-II improvements

**Insert under:** Results — HCO / Phase II.

> Absolute ALNS-HCO results are reported for Phase-I (ALNS-PR, 15 s), PhaseII-NoHCO, and PhaseII-Heterogeneous on the full Hanif panel (seeds 0–2). On the three audited aggregate metrics (fleet size, distance, charged energy), PhaseII-Heterogeneous matches Phase I in **350 of 360** matched (instance, TW, seed) cases (97.2%). The remaining 10 cases show aggregate improvement; none worsen those three metrics.
>
> Separating effects: PhaseI→PhaseII-NoHCO captures ordinary Phase-II search without the charging oracle (e.g. distance improved in 69/360 cases). PhaseII-NoHCO→PhaseII-Heterogeneous isolates the HCO-enabled channel and is mostly unchanged on aggregates, with some worsenings — Heterogeneous does not dominate NoHCO on this freeze. With `phase2_distance_tolerance` \(\varepsilon=0\), many destroy/repair neighbours are rejected by the distance gate even when HCO evaluations are feasible (`phase2_het_diagnosis/DIAGNOSIS.md`). Route-sequence identity with Phase I is **not** claimed (checksums absent from the publication CSV).

**Figure 4–5:** use NoHCO→HCO attribution and captions from `outputs/figures/final_figure_manifest.md`.

---

## F. HCO-Detour wording correction

**Replace any claim** that replacement / improve ranking is “pure detour throughout”.

> Under HCO-Detour, the station **shortlist** ranks candidates by insertion detour (and station id). The internal **replacement tournament** and non-worsening acceptance use a **geometry-led** key dominated by total route distance and charging-visit count, with insertion-arc detour only as a tertiary key. Preference decisions among geometrically ranked candidates remain invariant to permuting station prices and waits; feasibility replay still uses physical rates and waiting. See `HCO_DETOUR_VERIFICATION.md`.

---

## G. Schneider benchmark validation (including failures)

**Insert under:** External validation / Schneider results.

> We retain all **56** Schneider et al. large instances (`*_21`) and **336** ALNS attempts (FR and PR × seeds 0–2 × 30 s). Under the current solver and Euclidean-unrounded distance convention, **264** runs are feasible and **72** are infeasible (12 instances × 3 seeds × 2 methods). Primary FR/PR performance comparisons are **conditional on the 44 instances** feasible for both algorithms. On that panel, PR reduces charged energy (CI excludes 0) while the distance CI overlaps 0.
>
> A corrected route-level investigation using `replay_route` finds feasible station-insertion patterns for customers that fail direct depot–customer–depot walks. We therefore classify the 12 hard instances as **construction/search failures**, not as proven mathematical infeasibility of the official instances (alternative rounding conventions are not ruled out). Schneider stations are homogeneous with neutral price/wait placeholders; these runs validate routing and full vs partial recharging under equal budgets and **do not** support claims about heterogeneous price or waiting benefits.

**Files:** `schneider/table_main30_feasibility.csv`, `paired_main30_summaries.csv`, `feasibility_investigation_round4.json`, `NEUTRAL_CHARGING.md`.

---

## H. Economic and wage sensitivity — limitations

**Insert under:** Discussion / Economic interpretation (or Online appendix).

> Public charging expense in the main tables is the observed sum of price×en-route energy under the Hanif station model. Separate **post-hoc** scenarios apply hypothetical depot electricity tariffs \(\{0,0.10,0.20,0.50\}\) and wages per unit duration \(\{0,0.25,0.50,1.00\}\) without re-optimising routes (`economic_sensitivity/`). Totals are labelled as sensitivity analyses **not** optimised by ALNS/HCO. We do not claim that the algorithms minimise wage-inclusive or depot-tariff total cost.

---

## I. Terminology, units, equations, references (checklist)

| Item | Correction |
|---|---|
| Figure 5 | HCO vs NoHCO (family \(C_{ch}\)); **not** FR/PR |
| Figure 4 | Same comparison; metrics \(D, C_{ch}, T_{wait}, T_{ch}\) |
| “Detour-only HCO” | Shortlist yes; replacement = distance-led geometric |
| “Solved all Schneider” | **False** — 264/336 feasible |
| TW variants | Related benchmarks, not replications |
| Energy “total electricity” | Distinguish \(E_{\mathrm{enroute}}\), \(E_{\mathrm{depot}}\) (reconstructed), expense |
| Partial recharge benefit | Quantity (just-enough), not nonlinear charge speed |
| Equal budget | Wall-clock; report iteration disparity |
| \(\varepsilon\) | 0.0 as frozen config; not claimed optimal |
| BKS Schneider | Not claimed under unrounded Euclidean |

---

## J. Data Availability Statement (ready text)

> The ALNS-HCO and CARE-FI source code used in this study, frozen experiment configuration, and processed result tables are available in the project repository. Reviewer-revision canonical datasets are archived under `outputs/reviewer_revision/`, including the 288-run FR/PR convergence panel (`fr_pr_convergence_primary288/`), the 2520-row Hanif main table (`hanif_main_seeds012/`), Schneider 30 s feasibility and paired summaries (`schneider/`), HCO contribution audit (`hco_contribution_audit/`), and economic sensitivity tables (`economic_sensitivity/`). Raw publication solution rows reused for Hanif main reporting are under `outputs/raw/`. Benchmark instance files follow the Hanif and Schneider public distributions as documented in `dataset/` and converter metadata. Protected algorithmic modules on branch `reviewer-revision-2026` are verified identical to `main` for decision-logic files listed in `scripts/verify_no_algo_diff.py`.

---

## K. Declarations (separate required headings)

Use journal-required separate headings; paste and adapt:

### Declaration of competing interest  
> The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Funding  
> [Authors: insert grant numbers / agencies. Do not invent.]

### CRediT authorship contribution statement  
> [Authors: insert CRediT roles.]

### Acknowledgements  
> [Authors: insert as appropriate.]

### Data availability  
> See Data Availability Statement above (or journal “Data availability” heading pointing to the repository paths).

---

## L. Suggested table/figure inventory for the revision package

See `FINAL_TABLE_FIGURE_MANIFEST.csv` for paths. Minimum set for the response letter:

1. Convergence means & paired summaries (primary 288)  
2. Gap-change 30→300 summary  
3. Hanif method means + energy coverage  
4. HCO contribution counts  
5. Schneider feasibility + conditional paired FR/PR  
6. Economic sensitivity assumptions + selected scenario means  
7. Figures: FR/PR convergence; Hanif main; paper Figs 4–5 (NoHCO→HCO)  
