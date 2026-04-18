# Project evidence pack

This folder consolidates the most important, review-ready artifacts for project evaluation.

- autoPET main run: `autopet_fdg_full_post_best_dice_50epochs_20260324`
- autoPET comparison run: `autopet_fdg_full_50epochs_variant_comparison_20260324`
- Brain MRI backup run: `grenoble_gpu_20260324`
- includes run index: `yes`

## Key metrics

- autoPET mean Dice: `0.4867`
- autoPET mean FN volume (mL): `41.2100`
- autoPET mean FP volume (mL): `6.2934`
- Brain MRI accuracy: `0.8684`
- Brain MRI F1: `0.8889`
- Brain MRI ROC-AUC: `0.9391`

## XAI benchmark highlights

- autoPET top method: `integrated_gradients`
- Brain MRI top method: `integrated_gradients`

## Contents

- `autopet/`: metrics, comparison, optional method benchmark, selected figures
- `brain_mri/`: metrics, optional method benchmark, selected figures
- `traceability/`: requirement traceability map and optional run index snapshot
- `evidence_manifest.json`: explicit inventory of copied evidence files
- `INTERPRETATION.md`: concise interpretation blocks ready for report/slides
- `EVALUATION_ALIGNMENT.md`: rubric-oriented checklist for client/soutenance/plan-projet review
- `DEMO_RUNBOOK.md`: deterministic 2-3 minute demo flow aligned with `REQ-C2/C4/C5`
- `../brain_mri_refresh_xai_20260418/`: expanded qualitative Brain MRI XAI gallery (`16` balanced cases)
- `../brain_mri_xai_benchmark_20260418_clean_fast/`: clean-manifest sanity-check benchmark (fast protocol)

## Figure counts

- autoPET copied figures: `6`
- Brain MRI copied figures: `6`
