# Brain MRI XAI method benchmark

This file compares XAI methods with a protocol-level faithfulness metric.

Protocol:
- Selected test samples: `16`
- Max samples per class: `8`
- Fractions masked: `0.10, 0.20, 0.30`
- Bootstrap iterations: `2000`

Interpretation rule:
- `mean_confidence_drop` higher is better for this metric (attribution identifies pixels that matter more for model confidence).

## Ranking

- Rank 1: `integrated_gradients` -> `0.0877` (95% CI `0.0385` to `0.1354`)
- Rank 2: `occlusion` -> `0.0762` (95% CI `0.0449` to `0.1066`)
- Rank 3: `gradcam` -> `0.0195` (95% CI `0.0098` to `0.0298`)

## Notes for project interpretation

- This metric compares methods on model-faithfulness, not on medical truth by itself.
- Keep using lesion-level metrics and confusion matrix as the primary model-performance evidence.
- Use this benchmark to justify why one XAI method is preferred for discussion in the report.
