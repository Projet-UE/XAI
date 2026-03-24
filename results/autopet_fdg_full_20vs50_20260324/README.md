# autoPET FDG comparison snapshot

This folder compares two tracked autoPET FDG nnUNet runs on the same review split.

- Baseline: `nnUNetTrainer_20epochs` -> `autopet_fdg_full_20epochs_20260324`
- Candidate: `nnUNetTrainer_50epochs` -> `autopet_fdg_full_50epochs_20260324`
- Shared review cases: `7`

## Aggregate comparison

- Mean Dice: `0.1000 -> 0.3051` (`+0.2052`)
- Mean false negative volume (mL): `3.7056 -> 35.6684` (`+31.9628`)
- Mean false positive volume (mL): `386.5567 -> 30.4556` (`-356.1011`)
- Positive-case mean Dice: `0.2333 -> 0.3787` (`+0.1454`)
- Negative-case mean false positive volume (mL): `279.6244 -> 15.2745` (`-264.3499`)

See `comparison.json` for the per-case deltas.
