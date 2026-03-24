from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Occlusion
import matplotlib
import numpy as np
from PIL import Image
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .utils import ensure_dir, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std) + mean


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = denormalize(image_tensor.detach().cpu()).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def _normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array -= array.min()
    max_value = array.max()
    if max_value > 0:
        array /= max_value
    return array


def compute_attribution(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    method: str,
) -> np.ndarray:
    batch = input_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    target = 0

    if method == "gradcam":
        explainer = LayerGradCam(model, model.layer4[-1])
        attribution = explainer.attribute(batch, target=target)
        attribution = LayerAttribution.interpolate(attribution, batch.shape[-2:])
        heatmap = torch.relu(attribution.mean(dim=1)).squeeze(0).detach().cpu().numpy()
    elif method == "integrated_gradients":
        explainer = IntegratedGradients(model)
        attribution = explainer.attribute(batch, baselines=torch.zeros_like(batch), target=target, n_steps=25)
        heatmap = attribution.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
    elif method == "occlusion":
        explainer = Occlusion(model)
        attribution = explainer.attribute(
            batch,
            strides=(3, 8, 8),
            sliding_window_shapes=(3, 16, 16),
            baselines=0,
            target=target,
        )
        heatmap = attribution.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported XAI method: {method}")

    return _normalize_map(heatmap)


def save_explanation_panel(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    output_path: Union[str, Path],
    title: str,
) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)

    rgb_image = tensor_to_numpy_image(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title("Attribution")
    axes[2].imshow(rgb_image)
    axes[2].imshow(heatmap, cmap="magma", alpha=0.45)
    axes[2].set_title(title)

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def generate_explanations_for_loader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    methods: List[str],
    output_dir: Union[str, Path],
    max_samples_per_class: int = 2,
) -> Dict[str, Any]:
    target = ensure_dir(output_dir)
    model.to(device)
    model.eval()

    exported: List[Dict[str, Any]] = []
    seen_per_class: Dict[str, int] = {}

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].int().cpu().tolist()
        paths = batch["path"]
        class_names = batch["class_name"]

        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()

        for index in range(images.size(0)):
            class_name = class_names[index]
            seen_count = seen_per_class.get(class_name, 0)
            if seen_count >= max_samples_per_class:
                continue

            sample_name = Path(paths[index]).stem
            sample_dir = ensure_dir(target / class_name / sample_name)
            single_image = images[index].detach().cpu()

            methods_written: List[str] = []
            for method in methods:
                heatmap = compute_attribution(model, images[index], method)
                output_path = sample_dir / f"{method}.png"
                save_explanation_panel(single_image, heatmap, output_path, title=method)
                methods_written.append(method)

            exported.append(
                {
                    "path": paths[index],
                    "label": labels[index],
                    "class_name": class_name,
                    "probability": probabilities[index],
                    "methods": methods_written,
                }
            )
            seen_per_class[class_name] = seen_count + 1

        if all(count >= max_samples_per_class for count in seen_per_class.values()) and len(seen_per_class) >= 2:
            break

    report = {"exported": exported}
    save_json(report, target / "xai_summary.json")
    return report
