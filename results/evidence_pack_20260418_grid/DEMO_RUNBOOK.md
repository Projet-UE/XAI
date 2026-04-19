# Demo Runbook (2-3 minutes)

Use this script during soutenance/client review to cover the mandatory acceptance points
without running heavy training jobs live.

## Objective coverage

- `REQ-C2` (pipeline exécutable): show script interfaces and validated snapshots.
- `REQ-C4` (explications XAI): show attribution outputs and method benchmark files.
- `REQ-C5` (analyse critique): show tradeoff and interpretation files.

## Step-by-step sequence

### 1. Context and tracked runs (20-30s)
- Open `README.md` and state the tracked runs: `autopet_fdg_full_post_best_dice_50epochs_20260324`, `autopet_fdg_full_50epochs_variant_comparison_20260324`, `autopet_fdg_full_rebuild_best_label_50epochs_xai_3methods_20260419`, `grenoble_gpu_20260324`.
- Point to `traceability/requirement_traceability.json` for requirement coverage.

### 2. Core segmentation result and tradeoff (35-45s)
- Open `autopet/segmentation_metrics.json` (main result).
- Open `autopet/comparison.json` and explain best-Dice vs low-FP tradeoff.
- This covers the analysis expected by `REQ-C5`.

### 3. autoPET XAI evidence (30-40s)
- Open one or two files in `autopet/figures/`.
- Open `autopet/method_benchmark.json` and cite top method.
- If present, open `autopet/xai_segmentation_metrics.json` to show the benchmark's exact rebuilt-state metrics.
- Current top method: `integrated_gradients`.
- This covers explainability demonstration for `REQ-C4`.

### 4. Brain MRI backup evidence (25-35s)
- Open `brain_mri/metrics.json`.
- Open `brain_mri/xai_method_benchmark.json` when present.
- Current top method: `occlusion`.

### 5. Quick reproducibility proof (20-30s)
- Show validation commands (do not launch heavy training):

```bash
python scripts/validate_result_snapshot.py \
  --run-dir results/autopet_fdg_full_post_best_dice_50epochs_20260324 \
  --track autopet

python scripts/validate_result_snapshot.py \
  --run-dir results/grenoble_gpu_20260324 \
  --track brain_mri
```

These checks support `REQ-C2` by proving runnable, self-contained tracked outputs.

## Final one-line project message

autoPET FDG is the primary scientific line (segmentation + XAI tradeoff analysis), and Brain MRI is a reproducible backup line that confirms the XAI workflow on a second medical setting.
