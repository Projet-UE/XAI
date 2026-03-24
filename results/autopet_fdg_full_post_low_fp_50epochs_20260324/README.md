# autoPET FDG postprocessed 50-epoch snapshot (low false positives)

This folder tracks the `post_low_fp_50epochs` variant built on top of the same `fdg_full` autoPET FDG `nnUNetTrainer_50epochs` checkpoint, with `rank-mean_pet__minml-0p0__max-2` connected-component filtering.

- Split: `fdg_full`
- Mean Dice: `0.3743`
- Mean false negative volume (mL): `39.7864`
- Mean false positive volume (mL): `1.2708`
- Copied figures: `4`
