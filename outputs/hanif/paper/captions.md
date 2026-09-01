# Captions

Table 1. Experimental settings for the Hanif 1000-customer study (60 instances per TW variant, 5 seeds for stochastic methods). Lower is better for routing and charging metrics.

Table 2. Mean fleet size, distance, duration, and runtime. Stochastic methods are instance-level means over seeds, then averaged over instances.

Table 3. Paired ALNS-PR vs ALNS-HCO comparison. Relative change is the mean of instance-level $100(M_{HCO}-M_{PR})/M_{PR}$. Negative values are reductions. Wins/ties/losses use tolerance 1e-06. CI is a paired bootstrap 95% interval on that mean percentage. $p$ is Wilcoxon signed-rank on paired differences.

Table 4. Ablation: full recharge (ALNS-FR), partial recharge (ALNS-PR), HCO on frozen sequences (ALNS-PR+HCO), and the full method (ALNS-HCO).

Table 5. Mean paired percentage change of ALNS-HCO vs ALNS-PR by Solomon-style family.

Figure 1. ALNS-HCO architecture. Phase I is ALNS-PR. Phase II freezes $K=K^*$ and evaluates candidates with one Heterogeneous Charging Oracle.

Figure 2. Paired percentage changes of ALNS-HCO relative to ALNS-PR. The horizontal line is zero.

Figure 3. Instance-level charging expense: ALNS-HCO (y) vs ALNS-PR (x). Points below the diagonal are improvements.

Figure 4. Mechanism decomposition of charging-expense change (energy, expense/energy, wait/visit, time/energy, visits).

Figure 5. Family heatmap of mean paired percentage changes (ALNS-HCO vs ALNS-PR). Diverging scale centred at zero.

Figure 6. Mean runtime of equal-budget ALNS-PR versus ALNS-HCO Phase I / Phase II.

Figure S3. Runtime distribution by method.
