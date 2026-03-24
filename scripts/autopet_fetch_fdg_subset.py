#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autopet_xai.fetch import (
    DEFAULT_AUTOPET_FDG_META_URL,
    DEFAULT_AUTOPET_FDG_ZIP_URL,
    DEFAULT_AUTOPET_FILENAMES,
    extract_autopet_fdg_subset,
    load_autopet_metadata,
    select_autopet_fdg_cases,
    study_prefix_from_location,
)
from brain_tumor_xai.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a selective FDG-PET-CT-Lesions subset directly from the public autoPET NIfTI ZIP."
    )
    parser.add_argument("--destination-root", type=Path, required=True, help="Root where the selected study folders are materialized.")
    parser.add_argument("--zip-url", default=DEFAULT_AUTOPET_FDG_ZIP_URL)
    parser.add_argument("--metadata-url", default=DEFAULT_AUTOPET_FDG_META_URL)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--negative-count", type=int)
    parser.add_argument("--positive-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-path", type=Path, help="Where to save the selection report JSON.")
    parser.add_argument("--include-filenames", nargs="+", default=list(DEFAULT_AUTOPET_FILENAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination_root = ensure_dir(args.destination_root)
    metadata_rows = load_autopet_metadata(args.metadata_url)
    selected_rows = select_autopet_fdg_cases(
        metadata_rows,
        target_count=args.target_count,
        seed=args.seed,
        negative_count=args.negative_count,
        positive_count=args.positive_count,
    )
    selected_prefixes = [study_prefix_from_location(row["study_location"]) for row in selected_rows]
    extraction_report = extract_autopet_fdg_subset(
        zip_url=args.zip_url,
        selected_study_prefixes=selected_prefixes,
        destination_root=destination_root,
        include_filenames=args.include_filenames,
        overwrite=args.overwrite,
    )
    selection_report = {
        "zip_url": args.zip_url,
        "metadata_url": args.metadata_url,
        "target_count": args.target_count,
        "negative_count": sum(1 for row in selected_rows if row.get("diagnosis", "").upper() == "NEGATIVE"),
        "positive_count": sum(1 for row in selected_rows if row.get("diagnosis", "").upper() != "NEGATIVE"),
        "seed": args.seed,
        "destination_root": str(destination_root),
        "source_root_for_prepare": str(destination_root / "FDG-PET-CT-Lesions"),
        "include_filenames": args.include_filenames,
        "selected_cases": selected_rows,
        "extraction_report": extraction_report,
    }
    selection_path = args.selection_path or (destination_root / "fdg_subset_selection.json")
    save_json(selection_report, selection_path)


if __name__ == "__main__":
    main()
