from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from brain_tumor_xai.data import BrainTumorDataset, ensure_split_manifest
from brain_tumor_xai.evaluation import collect_predictions, compute_binary_classification_metrics
from brain_tumor_xai.model import build_resnet18_binary, load_checkpoint
from brain_tumor_xai.train import fit
from brain_tumor_xai.utils import select_device, set_seed
from brain_tumor_xai.xai import generate_explanations_for_loader


def test_training_eval_and_xai_end_to_end(tiny_dataset: Path, tmp_path: Path) -> None:
    set_seed(42)
    manifest = ensure_split_manifest(tiny_dataset, tmp_path / "split.json", seed=42)

    train_dataset = BrainTumorDataset(tiny_dataset, manifest, split="train", image_size=64, augment=True)
    val_dataset = BrainTumorDataset(tiny_dataset, manifest, split="val", image_size=64, augment=False)
    test_dataset = BrainTumorDataset(tiny_dataset, manifest, split="test", image_size=64, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model = build_resnet18_binary(pretrained=False)
    device = select_device("cpu")
    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        learning_rate=1e-3,
        output_dir=tmp_path / "training",
    )

    checkpoint_path = Path(result["best_checkpoint"])
    assert checkpoint_path.exists()

    reloaded_model = build_resnet18_binary(pretrained=False)
    load_checkpoint(reloaded_model, str(checkpoint_path), map_location=device)
    reloaded_model.to(device)

    labels, probabilities, _ = collect_predictions(reloaded_model, test_loader, device)
    metrics = compute_binary_classification_metrics(labels, probabilities)
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"} <= metrics.keys()

    xai_report = generate_explanations_for_loader(
        model=reloaded_model,
        dataloader=test_loader,
        device=device,
        methods=["gradcam", "integrated_gradients", "occlusion"],
        output_dir=tmp_path / "xai",
        max_samples_per_class=1,
    )
    assert xai_report["exported"]
    first_export = xai_report["exported"][0]
    for method in first_export["methods"]:
        image_path = tmp_path / "xai" / first_export["class_name"] / Path(first_export["path"]).stem / f"{method}.png"
        assert image_path.exists()
