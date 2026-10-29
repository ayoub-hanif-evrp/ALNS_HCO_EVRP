# Schneider neutral charging assumptions

Schneider et al. (2014) E-VRPTW instances are **homogeneous** in charging rate and do **not** define station waiting times or energy prices.

For CARE-FI / ALNS-HCO schema compatibility only:

| Attribute | Value | Rationale |
|---|---|---|
| Charging rate | `1/g` from the `.txt` vehicle line | Original homogeneous rate |
| Station waiting | disabled (`station_wait_expected=None` → wait = 0) | Original model has no wait |
| Expected price | `price_mu=0`, `price_sigma=0` → E[price]=1 | Uniform neutral placeholder |
| Heterogeneous stations | `False` | No invented station differentiation |

**Do not** use Schneider results to claim benefits of heterogeneous price or waiting. Schneider here validates **routing / full vs partial recharging** under equal budgets.

Distance convention: Euclidean unrounded; travel time = distance / `v`; energy = distance × `r` (as in the converter metadata). Published Schneider BKS may use different rounding or objective definitions — numerical BKS comparison is therefore **not** claimed.
