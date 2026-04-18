# Medical Imaging Baselines + XAI

This repository contains the modeling and explainability work for a medical imaging project built around two complementary tracks:

- **main track:** autoPET FDG PET/CT lesion segmentation + XAI
- **backup track:** brain MRI classification + XAI

The goal is not only to report model scores, but to build pipelines that are:

- reproducible
- scriptable
- interpretable
- directly reusable for the report, slides, and paper draft

All heavy runs were designed for **Grid'5000 Grenoble**, while Git only keeps lightweight tracked results under [`results/`](results/).

## Project at a glance

| Item | Choice |
| --- | --- |
| Main scientific direction | autoPET FDG PET/CT lesion segmentation + XAI |
| Backup direction | brain MRI classification + XAI |
| Main execution platform | Grid'5000 Grenoble |
| Main retained autoPET result | `post_best_dice_50epochs` |
| Main retained brain MRI result | Grenoble classification snapshot from `2026-03-24` |

## Latest review artifacts

If you need the quickest evaluator-facing entrypoint, start here:

- consolidated evidence pack:
  - [`results/evidence_pack_20260418_grid/README.md`](results/evidence_pack_20260418_grid/README.md)
- short interpretation blocks (report/slides ready):
  - [`results/evidence_pack_20260418_grid/INTERPRETATION.md`](results/evidence_pack_20260418_grid/INTERPRETATION.md)
- explicit evidence inventory:
  - [`results/evidence_pack_20260418_grid/evidence_manifest.json`](results/evidence_pack_20260418_grid/evidence_manifest.json)
- requirement traceability:
  - [`results/evidence_pack_20260418_grid/traceability/requirement_traceability.json`](results/evidence_pack_20260418_grid/traceability/requirement_traceability.json)
- frozen run index used in the pack:
  - [`results/evidence_pack_20260418_grid/traceability/run_index.json`](results/evidence_pack_20260418_grid/traceability/run_index.json)

For the detailed autoPET protocol benchmark (with paired CI deltas, failure taxonomy, and cross-method agreement):

- [`results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/method_benchmark.json`](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/method_benchmark.json)
- [`results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/method_benchmark.md`](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/method_benchmark.md)

For the refreshed Brain MRI protocol benchmark:

- [`results/brain_mri_xai_benchmark_20260418/xai_method_benchmark.json`](results/brain_mri_xai_benchmark_20260418/xai_method_benchmark.json)
- [`results/brain_mri_xai_benchmark_20260418/README.md`](results/brain_mri_xai_benchmark_20260418/README.md)

## Main track: autoPET FDG PET/CT segmentation + XAI

This is the line kept as the **primary project contribution** because it is the most coherent with the original topic: medical imaging, segmentation, and explainability.

### Setup

| Item | Choice |
| --- | --- |
| Dataset family | `autoPET I/II` |
| Data used in practice | `FDG-PET-CT-Lesions` |
| Scope used for the project | controlled FDG subset, not full raw data volume |
| Task | lesion segmentation |
| Baseline model | `nnUNet v2 (3d_fullres)` |
| Main checkpoint base | `fdg_full + nnUNetTrainer_50epochs` |
| Main retained variant | `post_best_dice_50epochs` |
| Secondary comparison | `post_low_fp_50epochs` |
| XAI methods | Saliency, Integrated Gradients, Occlusion |

### Main results

| Variant | Mean Dice | Mean FN volume | Mean FP volume |
| --- | --- | --- | --- |
| `raw_50epochs` | `0.3051` | `35.6684` mL | `30.4556` mL |
| `post_best_dice_50epochs` | `0.4867` | `41.2100` mL | `6.2934` mL |
| `post_low_fp_50epochs` | `0.3743` | `39.7864` mL | `1.2708` mL |

### Why this result is the one kept for the project

- `post_best_dice_50epochs` is the best overall segmentation result in the current project scope
- `post_low_fp_50epochs` is useful as a secondary comparison because it shows a more conservative low-false-positive regime
- together, these two variants give a clear and discussable tradeoff for the report and presentation

### How to read the XAI figures

In this repository, XAI should be read as:

- regions that influenced the model prediction more strongly
- not a direct claim that the highlighted area is automatically pathological

This is what makes the autoPET XAI analysis useful:

- on positive cases, it helps check whether the model focuses on lesion-related uptake
- on false positives, it helps explain why the model is attracted to non-lesion regions
- on difficult cases, it helps identify missed or diffuse attention patterns

### autoPET XAI gallery

The panels below combine:

- PET image
- CT image
- attribution overlay
- ground-truth mask
- predicted mask

#### Representative cases from the final all-review analysis

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_4848bebb10/integrated_gradients.png" width="230" alt="autoPET positive case PETCT_4848bebb10" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_be3e55a32f/integrated_gradients.png" width="230" alt="autoPET positive case PETCT_be3e55a32f" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_a1db71e797/integrated_gradients.png" width="230" alt="autoPET positive case PETCT_a1db71e797" />
</p>

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_05bed31780/integrated_gradients.png" width="230" alt="autoPET false positive case PETCT_05bed31780" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_402c061122/integrated_gradients.png" width="230" alt="autoPET hard negative case PETCT_402c061122" />
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_3bce0eb7aa/integrated_gradients.png" width="230" alt="autoPET true negative case PETCT_3bce0eb7aa" />
</p>

<p align="center">
  <img src="results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/figures/PETCT_e2309b8f92/integrated_gradients.png" width="230" alt="autoPET negative case PETCT_e2309b8f92" />
</p>

#### Variant comparison

<p align="center">
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_402c061122.png" width="220" alt="autoPET raw 50 epochs case PETCT_402c061122" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/raw_50epochs__PETCT_4848bebb10.png" width="220" alt="autoPET raw 50 epochs case PETCT_4848bebb10" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_best_dice_50epochs__PETCT_be3e55a32f.png" width="220" alt="autoPET best-Dice postprocessed case PETCT_be3e55a32f" />
  <img src="results/autopet_fdg_full_50epochs_variant_comparison_20260324/figures/post_low_fp_50epochs__PETCT_e2309b8f92.png" width="220" alt="autoPET low-FP postprocessed case PETCT_e2309b8f92" />
</p>

### Most useful autoPET files

- main result metrics:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_best_dice_50epochs_20260324/segmentation_metrics.json)
- secondary comparison:
  - [`results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json`](results/autopet_fdg_full_post_low_fp_50epochs_20260324/segmentation_metrics.json)
- comparison snapshot:
  - [`results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md`](results/autopet_fdg_full_50epochs_variant_comparison_20260324/README.md)
- all-review XAI interpretation:
  - [`results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json`](results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327/xai_analysis_summary.json)
- modeling handoff:
  - [`docs/autopet_modeling_handoff.md`](docs/autopet_modeling_handoff.md)

### Evaluation-grade XAI comparison (autoPET)

To satisfy a common-protocol XAI comparison requirement, run:

```bash
python scripts/autopet_analyze_xai.py \
  --review-cases-path artifacts/autopet_fdg_poc/fdg_full/xai/review_cases.json \
  --metrics-path artifacts/autopet_fdg_poc/fdg_full/review_metrics/metrics.json \
  --output-dir artifacts/autopet_fdg_poc/fdg_full/xai_analysis \
  --state-name post_best_dice_50epochs
```

This now exports:

- protocol-level method ranking with a composite score
- bootstrap 95% confidence intervals for key method metrics
- a markdown summary ready to cite in the report

For explicit paired method deltas (A-B) with confidence intervals:

```bash
python scripts/autopet_compare_xai_methods.py \
  --review-cases-path artifacts/autopet_fdg_poc/fdg_full/xai/review_cases.json \
  --metrics-path artifacts/autopet_fdg_poc/fdg_full/review_metrics/metrics.json \
  --output-dir artifacts/autopet_fdg_poc/fdg_full/xai_compare \
  --bootstrap-iterations 5000
```

Outputs:

- `method_benchmark.json`
- `method_benchmark.md`

## Backup track: Brain MRI classification + XAI

This track is kept as a **backup baseline**. It is simpler than the autoPET line, but it is still useful because it provides:

- a complete end-to-end classification baseline
- a lighter XAI setting
- a second experimental angle for discussion and comparison

### Setup

| Item | Choice |
| --- | --- |
| Dataset | Kaggle brain MRI tumor detection |
| Task | binary classification (`yes` / `no`) |
| Baseline model | `ResNet18` |
| Execution platform | Grid'5000 Grenoble |
| Tracked run length | `5` epochs |
| Image size | `224` |
| XAI methods | Grad-CAM, Integrated Gradients, Occlusion |

### Main results

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

<p align="center">
  <img src="results/grenoble_gpu_20260324/figures/metrics_overview.png" width="420" alt="Brain MRI metrics overview" />
  <img src="results/grenoble_gpu_20260324/figures/confusion_matrix.png" width="320" alt="Brain MRI confusion matrix" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/figures/sample_predictions.png" width="820" alt="Brain MRI tracked sample predictions" />
</p>

### Brain MRI XAI gallery

#### Integrated Gradients

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients positive case Y109" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/No22/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients negative case No22" />
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/integrated_gradients.png" width="420" alt="Brain MRI integrated gradients negative case 4 no" />
</p>

#### Grad-CAM

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/gradcam.png" width="280" alt="Brain MRI Grad-CAM positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/gradcam.png" width="280" alt="Brain MRI Grad-CAM positive case Y109" />
  <img src="results/grenoble_gpu_20260324/xai/no/No22/gradcam.png" width="280" alt="Brain MRI Grad-CAM negative case No22" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/gradcam.png" width="280" alt="Brain MRI Grad-CAM negative case 4 no" />
</p>

#### Occlusion

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/yes/Y195/occlusion.png" width="280" alt="Brain MRI Occlusion positive case Y195" />
  <img src="results/grenoble_gpu_20260324/xai/yes/Y109/occlusion.png" width="280" alt="Brain MRI Occlusion positive case Y109" />
  <img src="results/grenoble_gpu_20260324/xai/no/No22/occlusion.png" width="280" alt="Brain MRI Occlusion negative case No22" />
</p>

<p align="center">
  <img src="results/grenoble_gpu_20260324/xai/no/4%20no/occlusion.png" width="280" alt="Brain MRI Occlusion negative case 4 no" />
</p>

#### Method comparison on all tracked MRI samples

| Sample | Grad-CAM | Integrated Gradients | Occlusion |
| --- | --- | --- | --- |
| `yes/Y195.JPG` | ![](results/grenoble_gpu_20260324/xai/yes/Y195/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y195/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y195/occlusion.png) |
| `yes/Y109.JPG` | ![](results/grenoble_gpu_20260324/xai/yes/Y109/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y109/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/yes/Y109/occlusion.png) |
| `no/No22.jpg` | ![](results/grenoble_gpu_20260324/xai/no/No22/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/no/No22/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/no/No22/occlusion.png) |
| `no/4 no.jpg` | ![](results/grenoble_gpu_20260324/xai/no/4%20no/gradcam.png) | ![](results/grenoble_gpu_20260324/xai/no/4%20no/integrated_gradients.png) | ![](results/grenoble_gpu_20260324/xai/no/4%20no/occlusion.png) |

#### Extended Brain MRI XAI gallery (refresh run: 8 yes + 8 no)

The refreshed run below expands the examples to **16 test images** using the same model family (`ResNet18`) and two fast methods (`Grad-CAM`, `Integrated Gradients`) to provide broader qualitative coverage.

<p align="center">
  <img src="results/brain_mri_refresh_xai_20260418/yes/Y195/integrated_gradients.png" width="420" alt="Brain MRI refresh integrated gradients positive case Y195" />
  <img src="results/brain_mri_refresh_xai_20260418/yes/Y109/integrated_gradients.png" width="420" alt="Brain MRI refresh integrated gradients positive case Y109" />
</p>

<p align="center">
  <img src="results/brain_mri_refresh_xai_20260418/no/No22/integrated_gradients.png" width="420" alt="Brain MRI refresh integrated gradients negative case No22" />
  <img src="results/brain_mri_refresh_xai_20260418/no/4%20no/integrated_gradients.png" width="420" alt="Brain MRI refresh integrated gradients negative case 4 no" />
</p>

<p align="center">
  <img src="results/brain_mri_refresh_xai_20260418/yes/Y60/gradcam.png" width="420" alt="Brain MRI refresh Grad-CAM positive case Y60" />
  <img src="results/brain_mri_refresh_xai_20260418/no/no%2095/gradcam.png" width="420" alt="Brain MRI refresh Grad-CAM negative case no 95" />
</p>

### Most useful Brain MRI files

- tracked run summary:
  - [`results/grenoble_gpu_20260324/README.md`](results/grenoble_gpu_20260324/README.md)
- metrics:
  - [`results/grenoble_gpu_20260324/metrics.json`](results/grenoble_gpu_20260324/metrics.json)
- run configuration:
  - [`results/grenoble_gpu_20260324/run_config.json`](results/grenoble_gpu_20260324/run_config.json)
- actual XAI outputs:
  - [`results/grenoble_gpu_20260324/xai/`](results/grenoble_gpu_20260324/xai/)
- refreshed extended gallery (16 images):
  - [`results/brain_mri_refresh_xai_20260418/xai_summary.json`](results/brain_mri_refresh_xai_20260418/xai_summary.json)

### Evaluation-grade XAI comparison (Brain MRI)

To compare Brain MRI XAI methods with a quantitative faithfulness protocol:

```bash
python scripts/brain_mri_benchmark_xai_methods.py \
  --data-root "$HOME/data/brain-mri-images" \
  --manifest-path artifacts/training/splits/brain_mri_split.json \
  --checkpoint-path artifacts/training/checkpoints/best.pt \
  --output-dir artifacts/brain_mri_xai_benchmark \
  --max-samples-per-class 8
```

This benchmark ranks methods by predicted-class confidence drop after masking top-attribution pixels.
Higher confidence drop indicates stronger faithfulness for that protocol.

### Unified evidence pack for evaluation

Build one review-ready package with metrics, benchmark summaries, selected figures, and requirement traceability:

```bash
python scripts/build_project_evidence_pack.py \
  --results-root results \
  --run-index-path results/index.json \
  --autopet-main-run-id autopet_fdg_full_post_best_dice_50epochs_20260324 \
  --autopet-comparison-run-id autopet_fdg_full_50epochs_variant_comparison_20260324 \
  --brain-mri-run-id grenoble_gpu_20260324 \
  --autopet-xai-analysis-run-id autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327
```

### Frozen run index

A tracked immutable index of baseline run states is stored in:

- [`results/index.json`](results/index.json)

It includes run IDs, split hashes, checkpoint references, and script checksums.

### Snapshot completeness validation

To fail fast on incomplete tracked snapshots:

```bash
python scripts/validate_result_snapshot.py \
  --run-dir results/autopet_fdg_full_post_best_dice_50epochs_20260324 \
  --track autopet
```

For protocol-grade XAI evidence:

```bash
python scripts/validate_result_snapshot.py \
  --run-dir results/autopet_fdg_full_post_best_dice_50epochs_xai_allcases_20260327 \
  --track autopet \
  --require-protocol-benchmark
```

## Repository layout

```text
.
├── docs/
├── notebooks/
├── results/
├── scripts/
├── src/autopet_xai/
├── src/brain_tumor_xai/
└── tests/
```

Main components:

- `src/autopet_xai/`: PET/CT data preparation, nnUNet orchestration, metrics, and XAI
- `src/brain_tumor_xai/`: MRI data pipeline, classifier, evaluation, and XAI
- `scripts/`: command-line entry points for preparation, training, prediction, export, and analysis
- `results/`: lightweight tracked snapshots and figures kept in Git

## Running and reproducibility

The main execution logic is documented in:

- [`docs/grid5000.md`](docs/grid5000.md)
- [`docs/autopet_fdg.md`](docs/autopet_fdg.md)

Important project rules:

- do not commit raw datasets
- do not commit large checkpoints or bulky artifacts
- use the tracked `results/` snapshots for GitHub-facing summaries

## References

- Brain MRI dataset: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data>
- Brain MRI notebook reference: <https://www.kaggle.com/code/aayushontherocks/brain-tumor-xai-demo>
- autoPET code reference: <https://github.com/lab-midas/autoPET>
- autoPET FDG dataset reference: <https://doi.org/10.7937/gkr0-xv29>
