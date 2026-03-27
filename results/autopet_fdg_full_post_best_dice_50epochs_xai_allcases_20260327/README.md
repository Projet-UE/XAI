# autoPET FDG XAI analysis (best Dice, all review cases)

State analyzed: `post_best_dice_50epochs_allcases`

Reminder: XAI highlights which regions influenced the model, not which regions are automatically cancerous.

## Quantitative context

- Review case count: `7`
- Mean Dice: `0.4867`
- Mean false negative volume (mL): `41.2100`
- Mean false positive volume (mL): `6.2934`

## Review selection

- Available review cases: `7`
- Selected review cases: `7`
- Selected positives: `3`
- Selected negatives: `4`

## Case categories

- Positive detected: `3`
- Positive missed: `0`
- False positive: `2`
- True negative: `2`

## Preferred method summary: `integrated_gradients`

- On positive cases, `integrated_gradients` concentrates on average 0.025 of attribution mass inside the ground-truth lesion.
- On positive cases, attribution intensity is on average 4.58x higher inside the ground-truth lesion than outside it.
- On negative cases, `integrated_gradients` still allocates on average 0.011 of attribution mass inside predicted foreground, which helps explain residual false positives.
- On negative cases, attribution intensity inside predicted foreground is 4.30x the outside level on average.
- On explicitly false-positive cases, the same method allocates on average 0.022 of attribution mass inside predicted foreground rather than empty background.
- On false-positive cases, attribution intensity inside predicted foreground is 5.68x the outside level on average.

## Interpretation guidance

- In successful positive cases, we expect attribution to overlap lesion-related PET uptake and predicted lesion regions.
- In false positives, attribution can still focus on high-uptake but non-lesion regions, which helps explain over-segmentation.
- In missed positive cases, attribution may stay diffuse or shift away from part of the true lesion, which is consistent with false negatives.
