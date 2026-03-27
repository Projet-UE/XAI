# Medical Imaging Baselines + XAI

This repository now contains two complementary project tracks:

- a working **brain MRI classification + XAI** baseline used as a backup and quick experimentation path
- a new **PET/CT lesion segmentation + XAI** path built around the official autoPET FDG nnUNet baseline

The goal is to keep both pipelines reproducible and scriptable on Grid'5000, instead of relying on disconnected notebooks.

## At a glance

- primary project direction: **autoPET FDG PET/CT lesion segmentation + XAI**
- backup direction: **brain MRI classification + XAI**
- execution environment: **Grid'5000 Grenoble**
- final modeling base kept for the project: **`fdg_full` + `nnUNetTrainer_50epochs`**
- main presentation result: **`post_best_dice_50epochs`**

## What this repository contains

This repo is not just a code dump. It contains:

- reusable pipelines instead of one-off notebooks
- tracked lightweight experiment snapshots under [`results/`](results/)
- XAI visual examples that can directly be reused in reports and slides
- a clear separation between:
  - the **backup MRI classification baseline**
  - the **main PET/CT autoPET segmentation line**

## Project objective

The main goal of the project is to study **explainability in medical imaging**, not only to produce a prediction score.

In practice, that means:

- training a reproducible medical imaging model
- evaluating its predictions quantitatively
- generating XAI explanations
- understanding **why** the model succeeds or fails on representative cases

For the current project stage, we chose to focus on a **controlled FDG subset** instead of the full autoPET data volume, so that the pipeline remains executable, comparable, and interpretable within the project deadline.

## Project highlights

- primary track: **autoPET FDG PET/CT lesion segmentation + XAI**
- main result kept for the project: `post_best_dice_50epochs`
- secondary comparison: `post_low_fp_50epochs`
- broader XAI interpretation available on **all 7 review cases**
- backup track: **brain MRI classification + XAI**

## Current headline result

Main autoPET result (`post_best_dice_50epochs`, `fdg_full`, `nnUNetTrainer_50epochs`):

| Metric | Value |
| --- | --- |
| Mean Dice | `0.4867` |
| Mean false negative volume | `41.2100` mL |
| Mean false positive volume | `6.2934` mL |

Secondary comparison (`post_low_fp_50epochs`):

| Metric | Value |
| --- | --- |
| Mean Dice | `0.3743` |
| Mean false negative volume | `39.7864` mL |
| Mean false positive volume | `1.2708` mL |

## Experimental setup

### Main autoPET line

- dataset family: **autoPET I/II**
- data source used in practice: **FDG-PET-CT-Lesions**
- task: **lesion segmentation**
- baseline model: **nnUNet v2 (`3d_fullres`)**
- postprocessing explored: lightweight connected-component filtering
- XAI methods exported: **Saliency**, **Integrated Gradients**, **Occlusion**

### Backup MRI line

- dataset: Kaggle brain MRI tumor detection
- task: **binary classification**
- baseline model: **ResNet18**
- XAI methods: **Grad-CAM**, **Integrated Gradients**, **Occlusion**

## Main takeaways

- the raw `50 epochs` autoPET checkpoint was already usable, but postprocessing made it much more presentation-ready
- the retained main result, `post_best_dice_50epochs`, gives the best overall segmentation tradeoff for the project
- the low-FP variant is useful because it shows that the model behavior can be tuned depending on whether we prioritize Dice or false-positive suppression
- the XAI analysis is used here to explain **model behavior**, not to claim that highlighted regions are automatically pathological

## Visual examples

Main XAI examples from the final all-review snapshot:

| Strong positive | False positive | True negative |
| --- | --- | --- |
| ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_a1db71e797/integrated_gradients.png) | ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_05bed31780/integrated_gradients.png) | ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_3bce0eb7aa/integrated_gradients.png) |

These panels combine:

- original PET image with attribution overlay
- CT image with attribution overlay
- ground-truth mask
- predicted mask

This makes the repo immediately usable for discussing both successful detections and failure modes.

## More tracked figures

### Final all-review XAI gallery

| Positive case | Positive case | False positive | True negative |
| --- | --- | --- | --- |
| ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_4848bebb10/integrated_gradients.png) | ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_be3e55a32f/integrated_gradients.png) | ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_05bed31780/integrated_gradients.png) | ![](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_3bce0eb7aa/integrated_gradients.png) |

### Variant comparison snapshot

These tracked figures make the postprocessing tradeoff visible directly from GitHub:

| Raw 50 epochs | Raw 50 epochs | Best-Dice postprocess | Low-FP postprocess |
| --- | --- | --- | --- |
| ![](results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_402c061122.png) | ![](results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_4848bebb10.png) | ![](results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_best_dice_50epochs__PETCT_be3e55a32f.png) | ![](results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_low_fp_50epochs__PETCT_e2309b8f92.png) |

## How to read the XAI figures

The highlighted areas should be interpreted as:

- regions that influenced the model prediction more strongly
- not a direct medical proof that the highlighted area is a lesion

In this project, the XAI figures are used to answer questions such as:

- does the model focus on lesion-related uptake?
- does it highlight coherent regions in successful detections?
- does it drift toward irrelevant uptake or surrounding structures in false positives?
- does it miss part of the lesion in hard cases?

## References

- Dataset: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data>
- Notebook reference: <https://www.kaggle.com/code/aayushontherocks/brain-tumor-xai-demo>
- autoPET code reference: <https://github.com/lab-midas/autoPET>
- autoPET FDG dataset reference: <https://doi.org/10.7937/gkr0-xv29>

## Latest tracked run snapshots

Lightweight Grenoble run snapshots tracked in Git:

- Classification + XAI backup baseline:
  - `results/grenoble_gpu_20260324/README.md`
  - `results/grenoble_gpu_20260324/metrics.json`
  - `results/grenoble_gpu_20260324/run_config.json`
  - `results/grenoble_gpu_20260324/xai_samples.json`
- Current autoPET FDG reference checkpoint: `fdg_full` + `nnUNetTrainer_50epochs`
  - Raw baseline state `raw_50epochs`:
    - `results/autopet_fdg_full_50epochs_20260324/README.md`
    - `results/autopet_fdg_full_50epochs_20260324/segmentation_metrics.json`
    - `results/autopet_fdg_full_50epochs_20260324/run_config.json`
    - `results/autopet_fdg_full_50epochs_20260324/review_cases.json`
  - Primary postprocessed state `post_best_dice_50epochs`:
    - `results/autopet_fdg_full_post_best_dice_50epochs_20260324/README.md`
    - `results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json`
    - `results/autopet_fdg_full_post_best_dice_50epochs_20260324/run_config.json`
    - `results/autopet_fdg_full_post_best_dice_50epochs_20260324/review_cases.json`
  - Secondary postprocessed state `post_low_fp_50epochs`:
    - `results/autopet_fdg_full_post_low_fp_50epochs_20260324/README.md`
    - `results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json`
    - `results/autopet_fdg_full_post_low_fp_50epochs_20260324/run_config.json`
    - `results/autopet_fdg_full_post_low_fp_50epochs_20260324/review_cases.json`
  - Compact comparison snapshot:
    - `results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md`
    - `results/autopet_fdg_full_50epochs_variant_comparison_20260324/comparison.json`
    - `results/autopet_fdg_full_50epochs_variant_comparison_20260324/segmentation_metrics.json`
  - All-review XAI analysis on the primary postprocessed state:
    - `results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/README.md`
    - `results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json`
    - `results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/review_cases.json`

Tradeoff summary:
- `post_best_dice_50epochs` improves mean Dice over `raw_50epochs`, but raises mean false-negative volume relative to raw
- `post_low_fp_50epochs` suppresses mean false-positive volume much more aggressively than raw, with a smaller Dice gain than `post_best_dice_50epochs`
- The all-review XAI analysis gives a broader qualitative read on the primary result: 7 review cases, 3 positives, 4 negatives, with attribution intensity enriched inside lesions on positives and inside predicted foreground on false positives

Earlier exploratory autoPET milestones remain tracked for project history:
- `results/autopet_fdg_20260324/`
- `results/autopet_fdg_full_20260324/`
- `results/autopet_fdg_full_20epochs_20260324/`
- `results/autopet_fdg_full_comparison_20260324/`
- `results/autopet_fdg_full_20vs50_20260324/`
- `results/autopet_fdg_full_50epochs_postprocess_mean_pet_20260324/`

Project-ready handoff material:

- modeling handoff note: `docs/autopet_modeling_handoff.md`
- final figure/case selection: `results/autopet_fdg_final_selection_20260327.json`

This folder is meant to keep useful experiment metadata and summary outputs in Git, while the full `artifacts/` tree stays out of version control.

## Recommended files for the final project

If you only need the most useful material for the report, slides, or paper draft, start here:

- main result metrics:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json)
- secondary comparison:
  - [`results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json)
- compact comparison snapshot:
  - [`results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md`](results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md)
- broader XAI interpretation:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json`](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json)
- ready-to-use handoff note:
  - [`docs/autopet_modeling_handoff.md`](docs/autopet_modeling_handoff.md)

## Project layout

```text
.
├── docs/
├── notebooks/
├── scripts/
├── src/brain_tumor_xai/
└── tests/
```

Main components:

- `src/brain_tumor_xai/data.py`: scan the dataset, create reproducible splits, build datasets/loaders
- `src/brain_tumor_xai/model.py`: ResNet18 binary classifier
- `src/brain_tumor_xai/train.py`: training loop and checkpointing
- `src/brain_tumor_xai/evaluation.py`: metrics and reports
- `src/brain_tumor_xai/xai.py`: Grad-CAM, Integrated Gradients, Occlusion
- `scripts/train.py`: train the baseline
- `scripts/evaluate.py`: evaluate a trained checkpoint
- `scripts/generate_xai.py`: export XAI visualizations
- `src/autopet_xai/data.py`: FDG PET/CT normalization, manifests, splits, nnUNet dataset export
- `src/autopet_xai/fetch.py`: selective FDG subset download from the public autoPET NIfTI ZIP
- `src/autopet_xai/nnunet.py`: orchestration helpers for `nnUNetv2`
- `src/autopet_xai/metrics.py`: Dice and false-volume metrics for review cases
- `src/autopet_xai/xai.py`: qualitative PET/CT attribution export for lesion-centric review cases
- `scripts/autopet_prepare_fdg.py`: prepare the FDG dataset and create `fdg_dev` / `fdg_full`
- `scripts/autopet_fetch_fdg_subset.py`: fetch only the FDG cases needed for a first POC
- `scripts/autopet_train_nnunet.py`: plan/preprocess + train the official-style nnUNet baseline
- `scripts/autopet_predict_nnunet.py`: run review inference and export segmentation metrics
- `scripts/autopet_generate_xai.py`: export Saliency / Integrated Gradients / Occlusion figures
- `scripts/autopet_export_results.py`: create a lightweight Git-tracked snapshot under `results/`
- `scripts/autopet_analyze_xai.py`: summarize attribution behavior into a paper-friendly JSON/Markdown report
- `scripts/autopet_sweep_postprocess.py`: sweep lightweight connected-component filters on existing review predictions

## Grid'5000-only workflow

Important: this project is meant to be versioned in Git and executed on Grid'5000. Do not run training, evaluation, or XAI generation locally.

### 1. Start a Grenoble frontend session

From `https://intranet.grid5000.fr/notebooks`, launch a JupyterLab frontend with:

- `site=grenoble`
- `type=front`
- `project=irit`
- `queue=auto`

Use this frontend for bootstrap, Git operations, Kaggle download, and smoke checks.

### 2. Clone the repository on Grenoble

```bash
git config --global credential.https://github.com.useHttpPath true
git config --global user.name "Your GitHub Name"
git config --global user.email "your-email@example.com"

cd "$HOME"
git clone https://github.com/Projet-UE/XAI.git
cd XAI
git checkout feature/autopet-fdg-nnunet-baseline
git pull --ff-only
```

Pushes are done with HTTPS + token from the Grenoble session.

### 3. Create the environment on Grenoble

```bash
./scripts/grid5000_setup.sh
source .venv/bin/activate
```

If you prefer a requirements install:

```bash
pip install -r requirements.txt
```

### 4. Download and normalize the Kaggle dataset on Grenoble

Download with the Kaggle CLI inside the Grenoble session:

```bash
mkdir -p "$HOME/data"
kaggle datasets download \
  -d navoneel/brain-mri-images-for-brain-tumor-detection \
  -p "$HOME/data"
unzip "$HOME/data/brain-mri-images-for-brain-tumor-detection.zip" -d "$HOME/data/raw-brain-mri"
```

Normalize the extracted dataset so the final layout is:

```text
$HOME/data/brain-mri-images/
├── no/
└── yes/
```

The `data/` directory is ignored by Git on purpose.

### 5. Smoke checks on the Grenoble frontend

```bash
source .venv/bin/activate
pytest -q tests/test_data.py tests/test_pipeline.py
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/generate_xai.py --help
```

### 6. Start a Grenoble GPU session for heavy runs

From the same notebooks interface, launch a reserved-node Jupyter session with:

- `site=grenoble`
- `type=node`
- `project=irit`
- `queue=abaca`
- `oarres=/host=1/gpu=1`
- `walltime=04:00:00`

Use the same paths on Grenoble:

- repo: `$HOME/XAI`
- dataset: `$HOME/data/brain-mri-images`
- outputs: `$HOME/XAI/artifacts`

### 7. Train the baseline

```bash
source .venv/bin/activate
python scripts/train.py \
  --data-root "$HOME/data/brain-mri-images" \
  --artifacts-dir artifacts \
  --epochs 5 \
  --batch-size 8 \
  --image-size 224
```

This creates:

- a versioned split manifest in `artifacts/splits/brain_mri_split.json`
- training history in `artifacts/training/history.json`
- a checkpoint in `artifacts/training/checkpoints/best.pt`

### 8. Evaluate the checkpoint

```bash
source .venv/bin/activate
python scripts/evaluate.py \
  --data-root "$HOME/data/brain-mri-images" \
  --manifest-path artifacts/splits/brain_mri_split.json \
  --checkpoint-path artifacts/training/checkpoints/best.pt \
  --output-dir artifacts/evaluation
```

### 9. Generate explanations

```bash
source .venv/bin/activate
python scripts/generate_xai.py \
  --data-root "$HOME/data/brain-mri-images" \
  --manifest-path artifacts/splits/brain_mri_split.json \
  --checkpoint-path artifacts/training/checkpoints/best.pt \
  --output-dir artifacts/xai \
  --methods gradcam integrated_gradients occlusion \
  --max-samples-per-class 2
```

## Reproducibility

- fixed seed support across Python, NumPy, and PyTorch
- reproducible train/val/test split saved as JSON
- scripts export configs and metrics alongside results

## Detailed run guide

The main execution guide lives in `docs/grid5000.md`.
The PET/CT-specific guide lives in `docs/autopet_fdg.md`.

Do not commit raw datasets, large checkpoints, or bulk artifacts.
