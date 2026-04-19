# autoPET XAI method benchmark

State analyzed: `rebuild_best_label_50epochs_xai_3methods_20260419`

## Method ranking

- Rank 1: `integrated_gradients` (protocol score `0.0369`)
- Rank 2: `saliency` (protocol score `0.0124`)
- Rank 3: `occlusion` (protocol score `0.0103`)

## Paired bootstrap delta confidence intervals

Interpretation: positive `delta_a_minus_b` means method A is higher than method B for the given metric.

### `mass_ratio_inside_gt__positive_only`
- `integrated_gradients` - `occlusion`: delta `0.0355` with 95% CI [0.0000, 0.0597] on `3` paired cases.
- `integrated_gradients` - `saliency`: delta `0.0327` with 95% CI [0.0000, 0.0658] on `3` paired cases.
- `occlusion` - `saliency`: delta `-0.0028` with 95% CI [-0.0144, 0.0061] on `3` paired cases.

### `top10_ratio_inside_gt__positive_only`
- `integrated_gradients` - `occlusion`: delta `0.0063` with 95% CI [0.0000, 0.0096] on `3` paired cases.
- `integrated_gradients` - `saliency`: delta `0.0174` with 95% CI [0.0000, 0.0493] on `3` paired cases.
- `occlusion` - `saliency`: delta `0.0111` with 95% CI [-0.0062, 0.0396] on `3` paired cases.

### `mass_ratio_inside_prediction__false_positive_only`
- No valid paired comparisons available.

## Failure taxonomy and cross-method agreement

This section summarizes method behavior per case group and winner agreement rates.

- `positive_detected`: comparable cases `1`, unique-winner rate `1.0000`.
  winner share: integrated_gradients=1.0000, occlusion=0.0000, saliency=0.0000
- `positive_missed`: comparable cases `2`, unique-winner rate `0.5000`.
  winner share: integrated_gradients=0.6667, occlusion=0.1667, saliency=0.1667
- `false_positive`: comparable cases `0`, unique-winner rate `n/a`.
  winner share: integrated_gradients=n/a, occlusion=n/a, saliency=n/a
- `true_negative`: comparable cases `4`, unique-winner rate `0.0000`.
  winner share: integrated_gradients=0.3333, occlusion=0.3333, saliency=0.3333
