#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autopet_xai.nnunet import plan_and_preprocess, resolve_training_output_dir, train_model
from brain_tumor_xai.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/preprocess and train the nnUNet baseline for autoPET FDG.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--split-name", type=str, default="fdg_dev", choices=["fdg_dev", "fdg_full"])
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--skip-plan-and-preprocess", action="store_true")
    parser.add_argument("--disable-dataset-integrity-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = ensure_dir(args.artifacts_dir)
    split_specific_artifacts = artifacts_dir / args.split_name
    if not split_specific_artifacts.exists():
        split_specific_artifacts = artifacts_dir

    if not args.skip_plan_and_preprocess:
        plan_and_preprocess(
            dataset_id=args.dataset_id,
            artifacts_dir=split_specific_artifacts,
            verify_dataset_integrity=not args.disable_dataset_integrity_check,
        )
    train_model(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        artifacts_dir=split_specific_artifacts,
        trainer=args.trainer,
        plans=args.plans,
    )

    training_output_dir = resolve_training_output_dir(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        artifacts_dir=split_specific_artifacts,
        trainer=args.trainer,
        plans=args.plans,
    )
    save_json(
        {
            "artifacts_dir": str(split_specific_artifacts),
            "dataset_id": args.dataset_id,
            "split_name": args.split_name,
            "configuration": args.configuration,
            "fold": args.fold,
            "trainer": args.trainer,
            "plans": args.plans,
            "training_output_dir": str(training_output_dir),
        },
        split_specific_artifacts / "training_run_config.json",
    )


if __name__ == "__main__":
    main()
