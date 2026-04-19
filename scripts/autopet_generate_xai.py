#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autopet_xai.nnunet import resolve_training_output_dir
from autopet_xai.xai import generate_review_xai
from brain_tumor_xai.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative XAI exports for autoPET FDG review cases.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--split-name", type=str, default="fdg_dev", choices=["fdg_dev", "fdg_full"])
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-balanced-selection", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["saliency", "integrated_gradients", "occlusion"],
        choices=["saliency", "integrated_gradients", "occlusion"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = args.artifacts_dir / args.split_name if (args.artifacts_dir / args.split_name).exists() else args.artifacts_dir
    mapping = load_json(split_root / "manifests" / f"case_mapping_{args.split_name}.json")
    prediction_dir = args.prediction_dir or (split_root / "review_predictions")
    output_dir = ensure_dir(args.output_dir or split_root / "xai")
    training_output_dir = resolve_training_output_dir(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        artifacts_dir=split_root,
        trainer=args.trainer,
        plans=args.plans,
    )

    report = generate_review_xai(
        case_mapping=mapping,
        training_output_dir=training_output_dir,
        prediction_dir=prediction_dir,
        output_dir=output_dir,
        fold=args.fold,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        methods=args.methods,
        max_cases=args.max_cases,
        balance_classes=not args.disable_balanced_selection,
    )
    save_json(
        {
            "dataset_id": args.dataset_id,
            "configuration": args.configuration,
            "fold": args.fold,
            "trainer": args.trainer,
            "plans": args.plans,
            "checkpoint_name": args.checkpoint_name,
            "device": args.device,
            "training_output_dir": str(training_output_dir),
            "prediction_dir": str(prediction_dir),
            "output_dir": str(output_dir),
            "split_name": args.split_name,
            "methods": list(args.methods),
            "case_count": len(report["cases"]),
            "selected_case_ids": [case["case_id"] for case in report["cases"]],
            "balanced_selection": not args.disable_balanced_selection,
            "selection": report.get("selection", {}),
        },
        output_dir / "xai_run_config.json",
    )


if __name__ == "__main__":
    main()
