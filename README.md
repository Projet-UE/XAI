# Medical Imaging Baselines + XAI

This repository now contains two complementary project tracks:

- a working **brain MRI classification + XAI** baseline used as a backup and quick experimentation path
- a new **PET/CT lesion segmentation + XAI** path built around the official autoPET FDG nnUNet baseline

The goal is to keep both pipelines reproducible and scriptable on Grid'5000, instead of relying on disconnected notebooks.

## At a glance

- main scientific line: **autoPET FDG PET/CT lesion segmentation + XAI**
- backup line: **brain MRI classification + XAI**
- execution environment: **Grid'5000 Grenoble**
- main result currently retained for the project: **`post_best_dice_50epochs`**
- full tracked outputs live under [`results/`](results/)

## Repository overview

This repository is organized around **two complementary tracks**:

1. **autoPET FDG segmentation + XAI**
   - the main line kept for the project
   - based on PET/CT lesion segmentation
   - used to study both performance and model behavior through XAI
2. **Brain MRI classification + XAI**
   - a lighter backup baseline
   - useful for quick experiments and for comparison with a simpler setup

The objective is not only to report a score, but to build pipelines that are:

- reproducible
- scriptable
- comparable
- interpretable

For the current project stage, we intentionally focused on a **controlled subset** rather than the full raw data volume, so that the experiments remain feasible and analyzable within the project deadline.

## autoPET FDG PET/CT segmentation + XAI

This is the **main project track**.

### Why this is the main line

- it is closer to the original project direction in medical imaging
- it includes a genuine **segmentation** task rather than only classification
- it gives more interesting XAI behavior to analyze on successes and failures

### Experimental setup

| Item | Choice |
| --- | --- |
| Dataset family | `autoPET I/II` |
| Data used in practice | `FDG-PET-CT-Lesions` |
| Task | lesion segmentation |
| Baseline model | `nnUNet v2 (3d_fullres)` |
| Main checkpoint base | `fdg_full + nnUNetTrainer_50epochs` |
| Main retained variant | `post_best_dice_50epochs` |
| Secondary retained variant | `post_low_fp_50epochs` |
| XAI methods exported | Saliency, Integrated Gradients, Occlusion |
| Execution platform | Grid'5000 Grenoble |

### Main results

| Variant | Mean Dice | Mean FN volume | Mean FP volume |
| --- | --- | --- | --- |
| `raw_50epochs` | `0.3051` | `35.6684` mL | `30.4556` mL |
| `post_best_dice_50epochs` | `0.4867` | `41.2100` mL | `6.2934` mL |
| `post_low_fp_50epochs` | `0.3743` | `39.7864` mL | `1.2708` mL |

Interpretation:

- `post_best_dice_50epochs` is the best **overall project result**
- `post_low_fp_50epochs` is useful to show a more conservative regime with much lower false positives
- this gives a clean tradeoff story for the report and presentation

### What the XAI figures mean

In this project, XAI does **not** mean:

- “red = lesion”
- “highlighted area = medical truth”

Instead, it means:

- the model relied more strongly on that region to produce its prediction
- the figure helps explain **why** the model succeeded or failed

This is why the autoPET section is useful scientifically:

- on successful positives, we can check whether attention overlaps lesion-related uptake
- on false positives, we can see whether the model focuses on non-lesion uptake or irrelevant structures
- on hard cases, we can inspect whether the model misses part of the lesion or stays too diffuse

### autoPET XAI gallery

The panels below combine:

- original PET image with attribution overlay
- CT image with attribution overlay
- ground-truth mask
- predicted mask

#### Representative cases from the final all-review analysis

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_4848bebb10/integrated_gradients.png" width="230" alt="Positive autoPET case PETCT_4848bebb10" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_be3e55a32f/integrated_gradients.png" width="230" alt="Positive autoPET case PETCT_be3e55a32f" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_a1db71e797/integrated_gradients.png" width="230" alt="Positive autoPET case PETCT_a1db71e797" />
</p>

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_05bed31780/integrated_gradients.png" width="230" alt="False positive autoPET case PETCT_05bed31780" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_402c061122/integrated_gradients.png" width="230" alt="False positive or hard negative autoPET case PETCT_402c061122" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_3bce0eb7aa/integrated_gradients.png" width="230" alt="True negative autoPET case PETCT_3bce0eb7aa" />
</p>

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_e2309b8f92/integrated_gradients.png" width="230" alt="Negative autoPET case PETCT_e2309b8f92" />
</p>

#### Variant comparison figures

These tracked figures make the postprocessing tradeoff visible directly from GitHub:

<p align="center">
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_402c061122.png" width="220" alt="Raw 50 epochs case PETCT_402c061122" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_4848bebb10.png" width="220" alt="Raw 50 epochs case PETCT_4848bebb10" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_best_dice_50epochs__PETCT_be3e55a32f.png" width="220" alt="Best-Dice postprocessed case PETCT_be3e55a32f" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_low_fp_50epochs__PETCT_e2309b8f92.png" width="220" alt="Low-FP postprocessed case PETCT_e2309b8f92" />
</p>

### Most useful autoPET files for the project

- main result metrics:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json)
- secondary comparison:
  - [`results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json)
- comparison snapshot:
  - [`results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md`](results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md)
- broader XAI interpretation:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json`](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json)
- final case selection:
  - [`results/autopet_fdg_final_selection_20260327.json`](results/autopet_fdg_final_selection_20260327.json)
- handoff note:
  - [`docs/autopet_modeling_handoff.md`](docs/autopet_modeling_handoff.md)

## Brain MRI classification + XAI

This is the **backup track**. It is simpler than the autoPET line, but still useful because it gives:

- a complete end-to-end classification baseline
- a faster pipeline for quick checks
- a second XAI setting to compare with segmentation

### Experimental setup

| Item | Choice |
| --- | --- |
| Dataset | Kaggle brain MRI tumor detection |
| Task | binary classification (`yes` / `no`) |
| Baseline model | `ResNet18` |
| Execution platform | Grid'5000 Grenoble |
| Epochs in tracked run | `5` |
| Image size | `224` |
| XAI methods | Grad-CAM, Integrated Gradients, Occlusion |

### Tracked MRI classification result

| Metric | Value |
| --- | --- |
| Accuracy | `0.8684` |
| Precision | `0.9091` |
| Recall | `0.8696` |
| F1 | `0.8889` |
| ROC-AUC | `0.9391` |

Confusion matrix:

- true negatives: `13`
- false positives: `2`
- false negatives: `3`
- true positives: `20`

### Brain MRI tracked figures

The Brain MRI section now includes both tracked summary figures and the real XAI panels from the Grenoble run:

<p align="center">
  <img src="results/grenoble_gpu_20260324/figures/metrics_overview.png" width="420" alt="Brain MRI metrics overview" />
  <img src="results/grenoble_gpu_20260324/figures/confusion_matrix.png" width="320" alt="Brain MRI confusion matrix" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/figures/sample_predictions.png" width="820" alt="Brain MRI tracked sample predictions" />
</p>

### Brain MRI XAI gallery

#### Integrated Gradients examples

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients positive case Y109" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/No22/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients negative case No22" />
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients negative case 4 no" />
</p>

#### Grad-CAM examples

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/gradcam.png" width="280" alt="Brain MRI Grad-CAM positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/gradcam.png" width="280" alt="Brain MRI Grad-CAM positive case Y109" />
  <img src="results/grenoble_gpu_20260324/xai/no/No22/gradcam.png" width="280" alt="Brain MRI Grad-CAM negative case No22" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/gradcam.png" width="280" alt="Brain MRI Grad-CAM negative case 4 no" />
</p>

#### Occlusion examples

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/occlusion.png" width="280" alt="Brain MRI Occlusion positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/occlusion.png" width="280" alt="Brain MRI Occlusion positive case Y109" />
  <img src="results/grenoble_gpu_20260324/xai/no/No22/occlusion.png" width="280" alt="Brain MRI Occlusion negative case No22" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/occlusion.png" width="280" alt="Brain MRI Occlusion negative case 4 no" />
</p>

#### Method comparison on all tracked samples

| Sample | Grad-CAM | Integrated Gradients | Occlusion |
| --- | --- | --- | --- |
| `yes/Y195.JPG` | ![](results/grenoble_gpu_20260324/xai/yes/Y195/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y195/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y195/occlusion.png) |
| `yes/Y109.JPG` | ![](results/grenoble_gpu_20260324/xai/yes/Y109/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y109/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y109/occlusion.png) |
| `no/No22.jpg` | ![](results/grenoble_gpu_20260324/xai/no/No22/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/no/No22/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/no/No22/occlusion.png) |
| `no/4 no.jpg` | ![](results/grenoble_gpu_20260324/xai/no/4%20no/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/no/4%20no/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/no/4%20no/occlusion.png) |

### Tracked MRI sample predictions

The lightweight Git snapshot keeps sample metadata for 4 XAI examples:

| Ground truth | File | Predicted positive probability |
| --- | --- | --- |
| `yes` | `yes/Y195.JPG` | `0.9923` |
| `yes` | `yes/Y109.JPG` | `0.9074` |
| `no` | `no/No22.jpg` | `0.0324` |
| `no` | `no/4 no.jpg` | `0.0168` |

At the moment, the tracked MRI snapshot in Git keeps:

- metrics
- run configuration
- sample-level XAI metadata
- lightweight rendered figures built from the tracked snapshot
- actual tracked XAI panels for 4 representative samples

The Brain MRI section is still lighter than the autoPET line, but it now shows both visible summary results and the real XAI outputs directly on the repository front page.

### Most useful MRI files for the project

- tracked run summary:
  - [`results/grenoble_gpu_20260324/README.md`](results/grenoble_gpu_20260324/README.md)
- metrics:
  - [`results/grenoble_gpu_20260324/metrics.json`](results/grenoble_gpu_20260324/metrics.json)
- run configuration:
  - [`results/grenoble_gpu_20260324/run_config.json`](results/grenoble_gpu_20260324/run_config.json)
- tracked XAI sample metadata:
  - [`results/grenoble_gpu_20260324/xai_samples.json`](results/grenoble_gpu_20260324/xai_samples.json)
- actual XAI outputs:
  - [`results/grenoble_gpu_20260324/xai/`](results/grenoble_gpu_20260324/xai/)
- tracked figures:
  - [`results/grenoble_gpu_20260324/figures/metrics_overview.png`](results/grenoble_gpu_20260324/figures/metrics_overview.png)
  - [`results/grenoble_gpu_20260324/figures/confusion_matrix.png`](results/grenoble_gpu_20260324/figures/confusion_matrix.png)
  - [`results/grenoble_gpu_20260324/figures/sample_predictions.png`](results/grenoble_gpu_20260324/figures/sample_predictions.png)

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
