# autoPET FDG comparison snapshot

This folder compares two tracked autoPET FDG nnUNet runs on the same review split.

- Baseline: `nnUNetTrainer_10epochs` -> `autopet_fdg_full_20260324`
- Candidate: `nnUNetTrainer_20epochs` -> `autopet_fdg_full_20epochs_20260324`
- Shared review cases: `7`

## Aggregate comparison

- Mean Dice: `0.0944 -> 0.1000` (`+0.0056`)
- Mean false negative volume (mL): `11.1205 -> 3.7056` (`-7.4148`)
- Mean false positive volume (mL): `507.8887 -> 386.5567` (`-121.3319`)
- Positive-case mean Dice: `0.2202 -> 0.2333` (`+0.0131`)
- Negative-case mean false positive volume (mL): `529.7418 -> 279.6244` (`-250.1174`)

See `comparison.json` for the per-case deltas.
