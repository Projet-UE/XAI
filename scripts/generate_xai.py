#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from brain_tumor_xai.data import BrainTumorDataset, load_manifest
from brain_tumor_xai.model import build_resnet18_binary, load_checkpoint
from brain_tumor_xai.utils import select_device
from brain_tumor_xai.xai import generate_explanations_for_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XAI visualizations for a trained classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xai"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "integrated_gradients", "occlusion"],
        choices=["gradcam", "integrated_gradients", "occlusion"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    manifest = load_manifest(args.manifest_path)

    dataset = BrainTumorDataset(args.data_root, manifest, split="test", image_size=args.image_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_resnet18_binary(pretrained=False)
    load_checkpoint(model, str(args.checkpoint_path), map_location=device)
    generate_explanations_for_loader(
        model=model,
        dataloader=dataloader,
        device=device,
        methods=args.methods,
        output_dir=args.output_dir,
        max_samples_per_class=args.max_samples_per_class,
    )


if __name__ == "__main__":
    main()
