# Schneider ALNS-HCO vs literature

Self-contained experiment: **ALNS-HCO** on the 56 large Schneider EVRPTW
instances (100 customers, 21 stations), compared to published EVRPTW-PR
best-known solutions.

All code and all results for this experiment live in this folder.
Nothing is written under `outputs/` or `scripts/`.

## Method

Unchanged paper ALNS-HCO from `src/alns-hco`:

- Phase I: ALNS-PR (just-enough partial recharge), score `(unserved, K, distance)`
- Phase II: freeze `K = K*`, HCO on candidates, score `(distance, charging expense)`
- Literature ranking reported here: feasible, then **min K**, then **min distance**

BKS source: Erdelić & Carić (2022) Table 3 (EVRPTW-PR), taking the
lexicographic min of the literature column and Erdelić's own ALNS.

## Run

From the repository root:

```powershell
python schneider_alns_hco/run.py --quick --workers 8
python schneider_alns_hco/run.py --workers 8
python schneider_alns_hco/run.py --compile-only
```

Default budget: 90 s Phase I + 90 s Phase II, seeds `{0,1,2,3,4}`.
Per instance the table uses the lex-best seed.

```powershell
python schneider_alns_hco/run.py --workers 8 --phase1-s 300 --phase2-s 300
```

## Files

| Path | What |
|---|---|
| `run.py` | runner |
| `job.py` | one (instance, seed) solve |
| `load.py` | Schneider instance loader |
| `bks.py` | published PR BKS |
| `compare.py` | vs-literature tables |
| `config.py` | names, seeds, budget |
| `results/results.csv` | every run |
| `results/best_per_instance.csv` | lex-best seed per instance |
| `results/vs_literature.md` | comparison table |
| `results/summary.json` | headline counts |
| `results/routes/` | optional route dumps (`--save-routes`) |
