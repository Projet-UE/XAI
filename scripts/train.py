#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brain_tumor_xai.data import build_dataloaders, ensure_split_manifest
from brain_tumor_xai.model import build_resnet18_binary
from brain_tumor_xai.train import fit
from brain_tumor_xai.utils import ensure_dir, save_json, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a brain MRI classification baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    artifacts_dir = ensure_dir(args.artifacts_dir)
    manifest_path = args.manifest_path or artifacts_dir / "splits" / "brain_mri_split.json"
    manifest = ensure_split_manifest(
        args.data_root,
        manifest_path,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    dataloaders = build_dataloaders(
        args.data_root,
        manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet18_binary(pretrained=not args.no_pretrained)
    training_dir = artifacts_dir / "training"
    result = fit(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=training_dir,
    )

    save_json(
        {
            "data_root": str(args.data_root),
            "manifest_path": str(manifest_path),
            "classes": manifest["classes"],
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "pretrained": not args.no_pretrained,
            "best_checkpoint": result["best_checkpoint"],
        },
        training_dir / "run_config.json",
    )


if __name__ == "__main__":
    main()
