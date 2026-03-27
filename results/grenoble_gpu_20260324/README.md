# Grenoble GPU run snapshot — 2026-03-24

This folder stores a lightweight, tracked summary of one end-to-end run executed on Grid'5000 Grenoble.

It is intentionally small enough for Git:

- run configuration
- test metrics
- a summary of exported XAI samples

It does not contain the full raw artifacts tree, large checkpoints, or the dataset itself.

## Execution context

- Platform: Grid'5000 Grenoble
- Frontend host: `fgrenoble`
- GPU node used for the full run: `vercors16-1.grenoble.grid5000.fr`
- GPU: `NVIDIA L4`
- OAR job id: `2712949`
- Dataset root on Grenoble: `~/data/brain-mri-images`
- Full artifacts directory on Grenoble: `~/XAI/artifacts/grenoble_gpu_20260324`

## Pipeline scope

The run covered the full project pipeline:

1. train the binary classifier
2. evaluate the best checkpoint on the test split
3. generate XAI outputs with Grad-CAM, Integrated Gradients, and Occlusion

## Test metrics

- Accuracy: `0.8684`
- Precision: `0.9091`
- Recall: `0.8696`
- F1: `0.8889`
- ROC-AUC: `0.9391`
- Confusion matrix: `[[13, 2], [3, 20]]`

See [metrics.json](./metrics.json) for the tracked machine-readable summary.

## XAI summary

The run exported explanations for 4 test images:

- 2 `yes`
- 2 `no`

Methods:

- Grad-CAM
- Integrated Gradients
- Occlusion

The tracked summary is in [xai_samples.json](./xai_samples.json).

## Tracked figures

The snapshot now also includes lightweight visual summaries that are small enough to keep in Git:

- [metrics_overview.png](./figures/metrics_overview.png)
- [confusion_matrix.png](./figures/confusion_matrix.png)
- [sample_predictions.png](./figures/sample_predictions.png)
- real XAI panels under [`xai/`](./xai/)

### Metrics overview

![](./figures/metrics_overview.png)

### Confusion matrix

![](./figures/confusion_matrix.png)

### Sample predictions kept in the snapshot

![](./figures/sample_predictions.png)

## Tracked Brain MRI XAI panels

The actual XAI outputs from the Grenoble run are also tracked in Git for four representative samples:

- two `yes`
- two `no`

Methods:

- Grad-CAM
- Integrated Gradients
- Occlusion

### Integrated Gradients gallery

| Positive `Y195` | Positive `Y109` |
| --- | --- |
| ![](./xai/yes/Y195/integrated_gradients.png) | ![](./xai/yes/Y109/integrated_gradients.png) |

| Negative `No22` | Negative `4 no` |
| --- | --- |
| ![](./xai/no/No22/integrated_gradients.png) | ![](./xai/no/4%20no/integrated_gradients.png) |

### Grad-CAM gallery

| Positive `Y195` | Positive `Y109` |
| --- | --- |
| ![](./xai/yes/Y195/gradcam.png) | ![](./xai/yes/Y109/gradcam.png) |

| Negative `No22` | Negative `4 no` |
| --- | --- |
| ![](./xai/no/No22/gradcam.png) | ![](./xai/no/4%20no/gradcam.png) |

### Occlusion gallery

| Positive `Y195` | Positive `Y109` |
| --- | --- |
| ![](./xai/yes/Y195/occlusion.png) | ![](./xai/yes/Y109/occlusion.png) |

| Negative `No22` | Negative `4 no` |
| --- | --- |
| ![](./xai/no/No22/occlusion.png) | ![](./xai/no/4%20no/occlusion.png) |

### Method comparison for all tracked samples

| Sample | Grad-CAM | Integrated Gradients | Occlusion |
| --- | --- | --- | --- |
| `yes/Y195.JPG` | ![](./xai/yes/Y195/gradcam.png) | ![](./xai/yes/Y195/integrated_gradients.png) | ![](./xai/yes/Y195/occlusion.png) |
| `yes/Y109.JPG` | ![](./xai/yes/Y109/gradcam.png) | ![](./xai/yes/Y109/integrated_gradients.png) | ![](./xai/yes/Y109/occlusion.png) |
| `no/No22.jpg` | ![](./xai/no/No22/gradcam.png) | ![](./xai/no/No22/integrated_gradients.png) | ![](./xai/no/No22/occlusion.png) |
| `no/4 no.jpg` | ![](./xai/no/4%20no/gradcam.png) | ![](./xai/no/4%20no/integrated_gradients.png) | ![](./xai/no/4%20no/occlusion.png) |

## Notes

- This is a project-oriented baseline, not a claim of methodological novelty.
- The value kept in Git is the reproducible pipeline and a compact run snapshot.
- Full output folders remain outside Git by design.
