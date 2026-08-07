# Transformer Search-Move Guide P11 PASS (immutable)

Frozen after equal-budget ALNS validation on the 6 validation instances.

- Decision: **PASS**
- Method: Transformer search-move guide (portfolio ALNS: 50% unguided + 50% guided, lex-best)
- Warm start: Deterministic-FPS P10.5 PASS routes (`seed_0`)
- Holdouts: **not** evaluated
- Not node BC / not `bc_best_policy.pt`

## Gate
- `neural_no_regression_gate`: accepted (`quality_improved`)
- Neural participation on all 6 instances
- Accepted neural-guided moves > 0
- 6/6 feasible, 100% served

## Notable improvements vs full-budget unguided ALNS
| Instance | Base used | Portfolio used |
|---|---:|---:|
| c201_21 | 24 | **21** |
| r201_21 | 16 | **15** |

## How to enable in Deterministic-FPS
```yaml
hybrid_search:
  neural_guidance:
    enabled: true
    deprecated_bc_guide: false
    checkpoint: outputs/tgfps_p11/transformer_search_guide.pt
```

Do not overwrite. Working copies: `outputs/baselines/transformer_search_guide_p11/`, `outputs/tgfps_p11/`.
