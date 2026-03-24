# autoPET FDG 50-epoch variant comparison

This folder consolidates the tracked `fdg_full` autoPET FDG results around the same `nnUNetTrainer_50epochs` checkpoint.

States compared:
- `raw_50epochs`: raw review predictions
- `post_best_dice_50epochs`: `rank-mean_pet__minml-5p0__max-1`
- `post_low_fp_50epochs`: `rank-mean_pet__minml-0p0__max-2`

Key tradeoffs:
- raw `50 epochs`: mean Dice `0.3051`, mean FN `35.6684` mL, mean FP `30.4556` mL
- best-Dice postprocessing: mean Dice `0.4867`, mean FN `41.2100` mL, mean FP `6.2934` mL
- low-FP postprocessing: mean Dice `0.3743`, mean FN `39.7864` mL, mean FP `1.2708` mL

Interpretation:
- the best-Dice postprocessing variant improves mean Dice over raw, but increases false-negative volume relative to raw
- the low-FP variant suppresses false positives much more aggressively than raw, with a smaller Dice gain than the best-Dice variant
