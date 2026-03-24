# autoPET FDG 50-epoch post-processing sweep

This folder tracks a lightweight post-processing sweep on top of the `fdg_full` `50 epochs` raw predictions.

- Sweep family: `mean_pet` connected-component filtering
- Evaluated settings so far: `8` post-processed variants + raw baseline
- Current best by mean Dice: `rank-mean_pet__minml-5p0__max-1`
- Current low-false-positive candidate: `rank-mean_pet__minml-0p0__max-2`

## Raw 50-epoch baseline

- Mean Dice: `0.3051`
- Mean false negative volume (mL): `35.6684`
- Mean false positive volume (mL): `30.4556`

## Best mean-Dice candidate

- Label: `rank-mean_pet__minml-5p0__max-1`
- Mean Dice: `0.4867`
- Mean false negative volume (mL): `41.2100`
- Mean false positive volume (mL): `6.2934`
- Positive-case mean Dice: `0.4689`
- Negative-case mean false positive volume (mL): `9.1566`

## Low-false-positive candidate

- Label: `rank-mean_pet__minml-0p0__max-2`
- Mean Dice: `0.3743`
- Mean false negative volume (mL): `39.7864`
- Mean false positive volume (mL): `1.2708`
- Positive-case mean Dice: `0.5400`
- Negative-case mean false positive volume (mL): `0.3110`

See `summary.json` for the running ranking and `comparison.json` for raw-vs-postprocessed deltas.
