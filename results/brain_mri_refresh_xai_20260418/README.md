# Brain MRI Extended XAI Snapshot (2026-04-18)

This folder tracks an expanded qualitative XAI snapshot for the Brain MRI secondary line.

## What was generated

- Source model: `results/brain_mri_refresh_20260418` checkpoint
- Methods: `gradcam`, `integrated_gradients`
- Selection budget: `8` positive + `8` negative test images
- Output count: `33` files
  - `16` Grad-CAM figures
  - `16` Integrated Gradients figures
  - `xai_summary.json`

## Important data hygiene note

The split manifest used for this snapshot was regenerated with hidden-path filtering, so entries under `.ipynb_checkpoints` are excluded from the sampled cases.

## Why this snapshot is useful

- It increases visual coverage compared with the earlier 4-case gallery.
- It keeps the same pipeline and model family, so visual differences are easier to compare.
- It provides additional qualitative material for report/slides without committing large artifacts.

## Reproduction command (Grid'5000 Grenoble)

```bash
PYTHONPATH=/home/olmechi/XAI_exec/src python scripts/generate_xai.py \
  --data-root /home/olmechi/data/brain-mri-images \
  --manifest-path /home/olmechi/XAI_exec/artifacts/brain_mri_refresh_20260418/splits/brain_mri_split_clean2.json \
  --checkpoint-path /home/olmechi/XAI_exec/artifacts/brain_mri_refresh_20260418/training/checkpoints/best.pt \
  --output-dir /home/olmechi/XAI_exec/results/brain_mri_refresh_xai_20260418 \
  --device cpu \
  --max-samples-per-class 8 \
  --batch-size 4 \
  --methods gradcam integrated_gradients
```

## Main files

- `xai_summary.json`: exported file list with class labels and predicted probabilities
- `yes/*/*.png`: positive-case XAI overlays
- `no/*/*.png`: negative-case XAI overlays
