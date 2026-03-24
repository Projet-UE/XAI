# autoPET FDG + nnUNet + qualitative XAI

This guide describes the PET/CT segmentation track that complements the existing classification baseline.

## Goal

Build a first reproducible POC on Grid'5000 Grenoble using:

- dataset: `FDG-PET-CT-Lesions (TCIA)`
- model family: official autoPET `nnUNet` baseline
- target: segmentation metrics on a fixed review split plus qualitative XAI exports

## 1. Fetch a selective FDG NIfTI subset

Instead of downloading the full public `nifti.zip` archive (~283 GB), start with a selective subset pull directly from the remote ZIP:

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_fetch_fdg_subset.py \
  --destination-root "$HOME/data/autopet-fdg-source" \
  --target-count 64 \
  --seed 42
```

This materializes only the required study folders under:

```text
$HOME/data/autopet-fdg-source/FDG-PET-CT-Lesions/...
```

and writes a tracked selection report in:

```text
$HOME/data/autopet-fdg-source/fdg_subset_selection.json
```

If you already have TCIA DICOM instead, you can still convert it first with the official `lab-midas/autoPET` logic and then feed the resulting NIfTI layout to the preparation script below.

## 2. Prepare the normalized FDG dataset

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_prepare_fdg.py \
  --source-root "$HOME/data/autopet-fdg-source/FDG-PET-CT-Lesions" \
  --prepared-root "$HOME/data/autopet-fdg/prepared" \
  --artifacts-dir "$HOME/XAI/artifacts/autopet_fdg_poc" \
  --dataset-id 501 \
  --train-count 48 \
  --val-count 8 \
  --review-count 8 \
  --link-mode symlink
```

This creates:

- a normalized dataset under `$HOME/data/autopet-fdg/prepared`
- a global manifest under `artifacts/autopet_fdg_poc/manifests/`
- two split families:
  - `fdg_dev`
  - `fdg_full`
- split-specific nnUNet raw datasets under:
  - `artifacts/autopet_fdg_poc/fdg_dev/nnunet_raw/...`
  - `artifacts/autopet_fdg_poc/fdg_full/nnunet_raw/...`

## 3. Train the nnUNet baseline

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_train_nnunet.py \
  --artifacts-dir "$HOME/XAI/artifacts/autopet_fdg_poc" \
  --split-name fdg_dev \
  --dataset-id 501 \
  --configuration 3d_fullres \
  --fold 0
```

## 4. Predict on review cases and export segmentation metrics

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_predict_nnunet.py \
  --artifacts-dir "$HOME/XAI/artifacts/autopet_fdg_poc" \
  --split-name fdg_dev \
  --dataset-id 501 \
  --configuration 3d_fullres \
  --fold 0 \
  --device cuda
```

Tracked outputs:

- `review_metrics/segmentation_metrics.json`
- `review_metrics/per_case_metrics.json`

## 5. Generate qualitative XAI on review cases

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_generate_xai.py \
  --artifacts-dir "$HOME/XAI/artifacts/autopet_fdg_poc" \
  --split-name fdg_dev \
  --dataset-id 501 \
  --configuration 3d_fullres \
  --fold 0 \
  --device cuda \
  --methods saliency integrated_gradients occlusion \
  --max-cases 4
```

Tracked outputs:

- `xai/review_cases.json`
- `xai/<case_id>/saliency.png`
- `xai/<case_id>/integrated_gradients.png`
- `xai/<case_id>/occlusion.png`

## 6. Export a lightweight Git snapshot

```bash
cd "$HOME/XAI"
source .venv/bin/activate

python scripts/autopet_export_results.py \
  --artifacts-dir "$HOME/XAI/artifacts/autopet_fdg_poc" \
  --split-name fdg_dev \
  --run-id autopet_fdg_YYYYMMDD
```

This writes a small tracked folder under:

```text
results/autopet_fdg_YYYYMMDD/
```

## Notes

- Keep the classification branch as a backup and comparison point.
- The first target is a **solid FDG POC**, not a full challenge reproduction.
- The XAI step is qualitative by design in this first pass.
