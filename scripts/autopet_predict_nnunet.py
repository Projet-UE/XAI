#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autopet_xai.metrics import evaluate_review_predictions, save_segmentation_report
from autopet_xai.nnunet import predict_cases
from brain_tumor_xai.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nnUNet inference on the autoPET FDG review split and export metrics.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--split-name", type=str, default="fdg_dev", choices=["fdg_dev", "fdg_full"])
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = args.artifacts_dir / args.split_name if (args.artifacts_dir / args.split_name).exists() else args.artifacts_dir
    mapping = load_json(split_root / "manifests" / f"case_mapping_{args.split_name}.json")

    predictions_dir = ensure_dir(args.predictions_dir or split_root / "review_predictions")
    metrics_dir = ensure_dir(args.metrics_dir or split_root / "review_metrics")
    dataset_root = split_root / "nnunet_raw" / f"Dataset{args.dataset_id:03d}_AutoPETFDG"
    images_ts = dataset_root / "imagesTs"

    predict_cases(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        input_dir=images_ts,
        output_dir=predictions_dir,
        artifacts_dir=split_root,
        trainer=args.trainer,
        plans=args.plans,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )

    metrics = evaluate_review_predictions(mapping, prediction_dir=predictions_dir)
    save_segmentation_report(metrics, metrics_dir)
    save_json(
        {
            "predictions_dir": str(predictions_dir),
            "metrics_dir": str(metrics_dir),
            "split_name": args.split_name,
            "configuration": args.configuration,
            "fold": args.fold,
        },
        metrics_dir / "predict_run_config.json",
    )


if __name__ == "__main__":
    main()
