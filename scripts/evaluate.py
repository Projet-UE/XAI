#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brain_tumor_xai.data import BrainTumorDataset, load_manifest
from brain_tumor_xai.evaluation import collect_predictions, compute_binary_classification_metrics, save_evaluation_report
from brain_tumor_xai.model import build_resnet18_binary, load_checkpoint
from brain_tumor_xai.utils import select_device
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained brain MRI classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    manifest = load_manifest(args.manifest_path)

    dataset = BrainTumorDataset(args.data_root, manifest, split="test", image_size=args.image_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_resnet18_binary(pretrained=False)
    load_checkpoint(model, str(args.checkpoint_path), map_location=device)
    model.to(device)

    labels, probabilities, paths = collect_predictions(model, dataloader, device)
    metrics = compute_binary_classification_metrics(labels, probabilities)
    save_evaluation_report(metrics, labels, probabilities, paths, args.output_dir)


if __name__ == "__main__":
    main()
