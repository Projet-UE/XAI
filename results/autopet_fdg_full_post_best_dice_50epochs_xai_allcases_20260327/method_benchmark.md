# autoPET XAI method benchmark

State analyzed: `post_best_dice_50epochs_xai_allcases_20260327`

## Method ranking

- Rank 1: `integrated_gradients` (protocol score `0.0239`)

## Paired bootstrap delta confidence intervals

Interpretation: positive `delta_a_minus_b` means method A is higher than method B for the given metric.

### `mass_ratio_inside_gt__positive_only`
- No valid paired comparisons available.

### `top10_ratio_inside_gt__positive_only`
- No valid paired comparisons available.

### `mass_ratio_inside_prediction__false_positive_only`
- No valid paired comparisons available.

## Failure taxonomy and cross-method agreement

This section summarizes method behavior per case group and winner agreement rates.

- `positive_detected`: comparable cases `3`, unique-winner rate `1.0000`.
  winner share: integrated_gradients=1.0000
- `positive_missed`: comparable cases `0`, unique-winner rate `n/a`.
  winner share: integrated_gradients=n/a
- `false_positive`: comparable cases `2`, unique-winner rate `1.0000`.
  winner share: integrated_gradients=1.0000
- `true_negative`: comparable cases `2`, unique-winner rate `1.0000`.
  winner share: integrated_gradients=1.0000
