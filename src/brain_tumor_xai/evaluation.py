from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from .utils import ensure_dir, save_json


def compute_binary_classification_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = [1 if score >= threshold else 0 for score in probabilities]
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "threshold": threshold,
    }

    if len(set(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    else:
        metrics["roc_auc"] = None

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], list[str]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = batch["label"].to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            labels.extend(batch_labels.int().cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            paths.extend(batch["path"])

    return labels, probabilities, paths


def save_confusion_matrix_figure(confusion: list[list[int]], output_path: str | Path) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)

    matrix = np.array(confusion)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["no", "yes"])
    ax.set_yticks([0, 1], labels=["no", "yes"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_evaluation_report(
    metrics: dict[str, Any],
    labels: list[int],
    probabilities: list[float],
    paths: list[str],
    output_dir: str | Path,
) -> None:
    target = ensure_dir(output_dir)
    save_json(metrics, target / "metrics.json")
    rows = [
        {
            "path": path,
            "label": int(label),
            "probability": float(probability),
        }
        for path, label, probability in zip(paths, labels, probabilities, strict=True)
    ]
    save_json({"predictions": rows}, target / "predictions.json")
    save_confusion_matrix_figure(metrics["confusion_matrix"], target / "confusion_matrix.png")
