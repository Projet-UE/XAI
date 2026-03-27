#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render lightweight figures for the tracked Brain MRI results snapshot.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/grenoble_gpu_20260324"),
        help="Tracked results directory containing metrics.json and xai_samples.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_metrics_overview(metrics: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1.4]})
    figure.patch.set_facecolor("#f7f4ef")

    key_metrics = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    ]

    axes[0].axis("off")
    axes[0].set_facecolor("#f7f4ef")
    axes[0].text(0.0, 0.96, "Brain MRI classification", fontsize=17, fontweight="bold", color="#1f2937")
    axes[0].text(0.0, 0.86, "Tracked Grenoble snapshot", fontsize=11.5, color="#4b5563")
    axes[0].text(
        0.0,
        0.69,
        "This figure summarizes the backup classification baseline\nused alongside the main autoPET segmentation track.",
        fontsize=10.5,
        color="#374151",
        linespacing=1.45,
    )
    axes[0].text(0.0, 0.48, "Headline values", fontsize=12, fontweight="bold", color="#111827")

    y = 0.39
    for name, value in key_metrics:
        axes[0].text(0.02, y, f"{name}", fontsize=11, color="#374151")
        axes[0].text(0.64, y, f"{value:.4f}", fontsize=11, fontweight="bold", color="#111827")
        y -= 0.08

    axes[0].text(0.0, 0.03, "Model: ResNet18 | Task: yes/no tumor classification", fontsize=10, color="#6b7280")

    names = [name for name, _ in key_metrics]
    values = [value for _, value in key_metrics]
    colors = ["#0f766e", "#0f766e", "#0f766e", "#0f766e", "#1d4ed8"]

    axes[1].barh(names, values, color=colors, edgecolor="#0f172a")
    axes[1].set_xlim(0, 1.0)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Score")
    axes[1].set_title("Classification metrics", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.35)

    for index, value in enumerate(values):
        axes[1].text(min(value + 0.015, 0.96), index, f"{value:.3f}", va="center", fontsize=10, color="#111827")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    matrix = np.array(metrics["confusion_matrix"], dtype=np.int32)
    figure, ax = plt.subplots(figsize=(5.7, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Pred no", "Pred yes"]
    ax.set_xticks([0, 1], labels=classes)
    ax.set_yticks([0, 1], labels=["True no", "True yes"])
    ax.set_title("Brain MRI confusion matrix", fontsize=14, fontweight="bold")

    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value > threshold else "#111827"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=16, fontweight="bold")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground-truth label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_sample_predictions(samples_payload: dict, output_path: Path) -> None:
    samples = samples_payload["samples"]
    methods = ", ".join(samples_payload.get("methods", []))
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    figure.patch.set_facecolor("#f8fafc")

    for axis, sample in zip(axes.flatten(), samples):
        probability = float(sample["predicted_positive_probability"])
        truth = sample["label"]
        predicted = "yes" if probability >= 0.5 else "no"
        correct = truth == predicted

        if truth == "yes":
            base_color = "#fee2e2"
            accent_color = "#b91c1c"
        else:
            base_color = "#dcfce7"
            accent_color = "#166534"

        axis.set_facecolor(base_color)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

        axis.text(0.05, 0.89, sample["relative_path"], fontsize=10.5, fontweight="bold", color="#111827", wrap=True)
        axis.text(0.05, 0.69, f"True label: {truth}", fontsize=10.5, color="#374151")
        axis.text(0.05, 0.56, f"Predicted: {predicted}", fontsize=10.5, color="#374151")
        axis.text(0.05, 0.43, f"P(yes): {probability:.3f}", fontsize=10.5, color="#374151")
        axis.text(
            0.05,
            0.20,
            "Correct prediction" if correct else "Prediction error",
            fontsize=11,
            fontweight="bold",
            color=accent_color,
        )
        axis.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.05, color="#e5e7eb", transform=axis.transAxes, clip_on=False))
        axis.add_patch(
            plt.Rectangle(
                (0.05, 0.08),
                0.9 * probability,
                0.05,
                color="#2563eb" if predicted == "yes" else "#059669",
                transform=axis.transAxes,
                clip_on=False,
            )
        )

    figure.suptitle("Brain MRI tracked sample predictions", fontsize=15, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.02,
        f"Tracked XAI methods in the run: {methods}",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    figure.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    figures_dir = ensure_dir(results_dir / "figures")

    metrics = load_json(results_dir / "metrics.json")
    samples = load_json(results_dir / "xai_samples.json")

    plot_metrics_overview(metrics, figures_dir / "metrics_overview.png")
    plot_confusion_matrix(metrics, figures_dir / "confusion_matrix.png")
    plot_sample_predictions(samples, figures_dir / "sample_predictions.png")


if __name__ == "__main__":
    main()
