# Brain MRI XAI method benchmark

This file compares XAI methods with a protocol-level faithfulness metric.

Protocol:
- Selected test samples: `16`
- Max samples per class: `8`
- Fractions masked: `0.10, 0.20, 0.30`
- Bootstrap iterations: `500`

Interpretation rule:
- `mean_confidence_drop` higher is better for this metric (attribution identifies pixels that matter more for model confidence).

## Ranking

- Rank 1: `gradcam` -> `0.0253` (95% CI `0.0085` to `0.0460`)
- Rank 2: `integrated_gradients` -> `0.0181` (95% CI `-0.0384` to `0.0761`)

## Notes for project interpretation

- This metric compares methods on model-faithfulness, not on medical truth by itself.
- Keep using lesion-level metrics and confusion matrix as the primary model-performance evidence.
- Use this benchmark to justify why one XAI method is preferred for discussion in the report.
