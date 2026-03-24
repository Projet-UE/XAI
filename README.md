# Medical Imaging Baselines + XAI

This repository now contains two complementary project tracks:

- a working **brain MRI classification + XAI** baseline used as a backup and quick experimentation path
- a new **PET/CT lesion segmentation + XAI** path built around the official autoPET FDG nnUNet baseline

The goal is to keep both pipelines reproducible and scriptable on Grid'5000, instead of relying on disconnected notebooks.

## References

- Dataset: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data>
- Notebook reference: <https://www.kaggle.com/code/aayushontherocks/brain-tumor-xai-demo>
- autoPET code reference: <https://github.com/lab-midas/autoPET>
- autoPET FDG dataset reference: <https://doi.org/10.7937/gkr0-xv29>

## Latest tracked run snapshots

Lightweight Grenoble run snapshots tracked in Git:

- Classification + XAI baseline:
  - `results/grenoble_gpu_20260324/README.md`
  - `results/grenoble_gpu_20260324/metrics.json`
  - `results/grenoble_gpu_20260324/run_config.json`
  - `results/grenoble_gpu_20260324/xai_samples.json`
- autoPET FDG nnUNet + qualitative XAI POC:
  - `results/autopet_fdg_20260324/README.md`
  - `results/autopet_fdg_20260324/segmentation_metrics.json`
  - `results/autopet_fdg_20260324/run_config.json`
  - `results/autopet_fdg_20260324/review_cases.json`

This folder is meant to keep useful experiment metadata and summary outputs in Git, while the full `artifacts/` tree stays out of version control.

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
