# autoPET Modeling Handoff

This note is the final handoff for the modeling/XAI part of the project.

## What To Keep As The Main Result

Use the following result as the main autoPET outcome:

- state: `post_best_dice_50epochs`
- split: `fdg_full`
- model: `nnUNetTrainer_50epochs`
- configuration: `3d_fullres`
- snapshot: [results/autopet_fdg_full_post_best_dice_50epochs_20260324/README.md](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_20260324/README.md)

Main metrics:

- mean Dice: `0.4867`
- mean false negative volume: `41.2100` mL
- mean false positive volume: `6.2934` mL

Use the following secondary comparison:

- state: `post_low_fp_50epochs`
- snapshot: [results/autopet_fdg_full_post_low_fp_50epochs_20260324/README.md](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_low_fp_50epochs_20260324/README.md)

Secondary metrics:

- mean Dice: `0.3743`
- mean false negative volume: `39.7864` mL
- mean false positive volume: `1.2708` mL

Interpretation:

- `post_best_dice_50epochs` is the main reference result because it gives the strongest Dice.
- `post_low_fp_50epochs` is useful to discuss the false-positive tradeoff.

## What To Say About XAI

Use the all-review analysis here:

- summary: [results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/README.md](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/README.md)
- machine-readable report: [results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json)

Safe interpretation:

- XAI does not indicate that a highlighted region is automatically cancerous.
- XAI shows which regions influenced the model output.
- On this review set, attribution is more intense inside lesion regions on positive cases than outside them.
- On false-positive cases, attribution is more intense inside predicted foreground than outside it, which supports the interpretation that the model is reacting to non-lesion but model-salient uptake regions.

Paper-safe findings from the current result:

- `7` review cases were analyzed with XAI on the primary postprocessed result.
- case categories: `3` positive detected, `2` false positive, `2` true negative
- preferred method for discussion: `integrated_gradients`
- on positive cases, attribution intensity is about `4.58x` higher inside the ground-truth lesion than outside
- on negative cases, attribution intensity inside predicted foreground is about `4.30x` the outside level
- on explicit false positives, attribution intensity inside predicted foreground is about `5.68x` the outside level

## What To Show In The Report Or Slides

Recommended final case set:

1. strong positive case:
   - case: `PETCT_a1db71e797`
   - category: `positive_detected`
   - Dice: `0.8983`
   - figure: [integrated_gradients.png](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_a1db71e797/integrated_gradients.png)

2. moderate positive case:
   - case: `PETCT_be3e55a32f`
   - category: `positive_detected`
   - Dice: `0.2979`
   - figure: [integrated_gradients.png](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_be3e55a32f/integrated_gradients.png)

3. false-positive case:
   - case: `PETCT_05bed31780`
   - category: `false_positive`
   - false-positive volume: `8.5843` mL
   - figure: [integrated_gradients.png](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_05bed31780/integrated_gradients.png)

4. true-negative case:
   - case: `PETCT_3bce0eb7aa`
   - category: `true_negative`
   - figure: [integrated_gradients.png](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_3bce0eb7aa/integrated_gradients.png)

Optional extra false-positive example:

- case: `PETCT_402c061122`
- false-positive volume: `28.0421` mL
- figure: [integrated_gradients.png](/home/arman/Bureau/f/XAI/results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_402c061122/integrated_gradients.png)

## Suggested Result Paragraph

Suggested wording:

Our main segmentation result is the `post_best_dice_50epochs` variant built from the `fdg_full` autoPET FDG subset and the `nnUNetTrainer_50epochs` baseline. This configuration reaches a mean Dice of `0.4867`, with a mean false negative volume of `41.2100` mL and a mean false positive volume of `6.2934` mL on the review set. Compared with the raw `50 epochs` predictions, this postprocessed variant improves Dice substantially while reducing false positives.

Suggested XAI wording:

Qualitative XAI analysis was conducted with `integrated_gradients` on all `7` review cases of the main result. The analysis shows that, on positive cases, attribution intensity is higher inside lesion regions than outside them, whereas on false-positive cases attribution remains concentrated inside predicted foreground regions despite the absence of ground-truth lesions. This suggests that the model relies on lesion-related PET/CT patterns in successful cases, but can also be attracted by non-lesion uptake patterns in failure cases.

## Suggested Limitations Paragraph

- the experiments were performed on a controlled subset, not the full TCIA FDG dataset
- the XAI study is qualitative and should not be interpreted as a clinical validation
- highlighted regions show model influence, not direct medical truth
- the current conclusions are about model behavior on this subset and setup

## What Not To Spend Time On Anymore

- do not launch new heavy training runs
- do not scale to the full `~400 GB` dataset
- do not open PSMA / multi-tracer work now
- do not change the main result again unless a supervisor explicitly asks

## Final Position

For the modeling part, this is enough to freeze:

- one main result
- one secondary tradeoff result
- one broader XAI analysis across all review cases
- one ready-to-use interpretation for the report or paper
