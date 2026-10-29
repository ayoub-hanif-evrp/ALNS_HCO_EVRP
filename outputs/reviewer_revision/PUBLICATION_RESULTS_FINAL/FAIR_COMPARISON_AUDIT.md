# FR versus PR — experimental fairness audit

## Stacks
- **ALNS-FR:** GI (greedy) initialization + care-fi ALNS with full-recharge station policy
- **ALNS-PR:** alns_hco Phase-I ALNS-PR with just-enough charging (no separate GI stage in PR path)

## Budgets
- Nominal ALNS search limits matched for paired FR/PR studies: **True**
- End-to-end including GI: FR runtime includes GI; PR runtime is Phase-I ALNS only — report both
- Iteration throughput equal: **False**
- Mean iterations (primary288): `{'ALNS-FR|30s': 3897.3541666666665, 'ALNS-FR|120s': 15954.666666666666, 'ALNS-FR|300s': 44398.270833333336, 'ALNS-PR|30s': 1146.75, 'ALNS-PR|120s': 4800.5625, 'ALNS-PR|300s': 12156.083333333334}`

## Isolation
Isolated charging-policy-only comparison: **False**.
Primary FR vs PR compares two search configurations (initialization + recharge policy + operators), not a perfectly isolated charging-policy intervention. Fixed-route HCO contrasts isolate station choice on identical Phase-I customer routes.

Fixed-route Detour/Heterogeneous and the recharge probe table isolate charging decisions on shared Phase-I customer routes without claiming FR≡PR operators.
