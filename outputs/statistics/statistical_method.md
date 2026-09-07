# Statistical method

1. For stochastic methods, average the five seeds per (instance, TW, method).
2. Paired comparisons use the 60 instances within each TW regime that have both methods.
3. Relative change for metric M is the **mean of instance-level** percentages
   \(100(M_B - M_A)/M_A\). Negative = reduction.
4. 95% CI: paired bootstrap of that mean percentage (seed=0).
5. Wins/ties/losses: lower is better, tolerance 1e-6.
6. Wilcoxon signed-rank on absolute paired differences.
7. Aggregate ratio-of-means percentages are **not** mixed into the mean-paired-% columns.
