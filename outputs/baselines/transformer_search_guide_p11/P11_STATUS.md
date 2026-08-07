# P11 status — **PASS**

## What shipped
| Piece | Path |
|---|---|
| Model | `src/models/search_move_guide.py` |
| Guide wrapper | `src/search/transformer_search_guide.py` |
| ALNS hooks + move logger | `src/search/alns.py` |
| Det-FPS portfolio ALNS | `src/search/deterministic_fps.py` |
| Collect | `scripts/collect_search_move_demos.py` |
| Train | `scripts/train_transformer_search_guide.py` |
| Validate | `scripts/run_p11_search_guide_validation.py` |
| Tests | `tests/test_search_move_guide.py` |
| Checkpoint | `outputs/tgfps_p11/transformer_search_guide.pt` |
| Freeze | `outputs/baselines/transformer_search_guide_p11_pass/` |

## Decision
**PASS** — portfolio (unguided 50% + guided 50%, lex-best) improves aggregate quality vs full-budget unguided ALNS on val-6, with neural participation and no feasibility regression.

## Notes
- Guide ranks ALNS operators / removals / routes; exact replay remains the oracle.
- BC construction checkpoints are rejected by filename and `deprecated_bc_guide`.
- Holdouts never used in collect/train/eval.
