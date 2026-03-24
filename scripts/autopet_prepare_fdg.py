#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autopet_xai.data import build_fdg_manifest, create_versioned_fdg_splits, export_nnunet_dataset
from brain_tumor_xai.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FDG-PET-CT-Lesions data for the autoPET nnUNet baseline.")
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing one folder per case with PET/CT/label volumes.")
    parser.add_argument("--prepared-root", type=Path, required=True, help="Normalized output directory with one case per folder.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--review-count", type=int, default=8)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = ensure_dir(args.artifacts_dir)
    manifests_dir = ensure_dir(artifacts_dir / "manifests")
    split_dir = ensure_dir(artifacts_dir / "splits")

    manifest = build_fdg_manifest(
        source_root=args.source_root,
        prepared_root=args.prepared_root,
        manifest_path=manifests_dir / "fdg_manifest.json",
        seed=args.seed,
        link_mode=args.link_mode,
    )
    split_records = create_versioned_fdg_splits(
        manifest,
        split_dir=split_dir,
        seed=args.seed,
        dev_counts=(args.train_count, args.val_count, args.review_count),
    )

    exported_splits = {}
    for split_name, split_record in split_records.items():
        split_artifacts_dir = ensure_dir(artifacts_dir / split_name)
        split_manifests_dir = ensure_dir(split_artifacts_dir / "manifests")
        split_splits_dir = ensure_dir(split_artifacts_dir / "splits")
        save_json(split_record, split_splits_dir / f"{split_name}.json")
        save_json({"parent_manifest": str(manifests_dir / "fdg_manifest.json")}, split_manifests_dir / "manifest_pointer.json")
        exported_splits[split_name] = export_nnunet_dataset(
            manifest=manifest,
            split_record=split_record,
            dataset_id=args.dataset_id,
            nnunet_raw_root=split_artifacts_dir / "nnunet_raw",
            link_mode=args.link_mode,
        )
        save_json(exported_splits[split_name], split_manifests_dir / f"case_mapping_{split_name}.json")
        save_json(
            {
                "dataset_id": args.dataset_id,
                "manifest_path": str(manifests_dir / "fdg_manifest.json"),
                "split_path": str(split_splits_dir / f"{split_name}.json"),
                "case_mapping_path": str(split_manifests_dir / f"case_mapping_{split_name}.json"),
                "prepared_root": str(args.prepared_root),
            },
            split_artifacts_dir / "prepare_report.json",
        )

    save_json(
        {
            "dataset_id": args.dataset_id,
            "manifest_path": str(manifests_dir / "fdg_manifest.json"),
            "prepared_root": str(args.prepared_root),
            "splits": {name: str(split_dir / f"{name}.json") for name in split_records},
            "case_mappings": {
                name: str((artifacts_dir / name / "manifests" / f"case_mapping_{name}.json")) for name in split_records
            },
        },
        artifacts_dir / "prepare_report.json",
    )


if __name__ == "__main__":
    main()
