from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import SimpleITK as sitk

from brain_tumor_xai.utils import ensure_dir, load_json, save_json

VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd")
LinkMode = Literal["copy", "symlink"]


def _is_volume_file(path: Path) -> bool:
    return any(str(path).lower().endswith(suffix) for suffix in VOLUME_EXTENSIONS)


def _normalize_case_id(raw_case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_case_id.strip()).strip("_")
    return cleaned or "case"


def _keyword_score(path: Path, keywords: Sequence[str]) -> int:
    name = path.name.lower()
    return sum(1 for keyword in keywords if keyword in name)


def _pick_best_candidate(candidates: Sequence[Path], keywords: Sequence[str], label: str, case_dir: Path) -> Path:
    if not candidates:
        raise ValueError(f"Could not identify a {label} file inside {case_dir}")
    ranked = sorted(candidates, key=lambda candidate: (_keyword_score(candidate, keywords), str(candidate)), reverse=True)
    best = ranked[0]
    if _keyword_score(best, keywords) == 0 and len(candidates) > 1:
        raise ValueError(
            f"Ambiguous {label} candidates in {case_dir}. Please rename files with clear modality keywords or pre-normalize them."
        )
    return best


def _read_volume_metadata(volume_path: Union[str, Path]) -> Dict[str, Any]:
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayViewFromImage(image)
    return {
        "shape_zyx": [int(dim) for dim in array.shape],
        "spacing_xyz": [float(value) for value in image.GetSpacing()],
        "origin_xyz": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
        "lesion_positive": bool(array.max() > 0),
        "positive_voxel_count": int((array > 0).sum()),
    }


def discover_fdg_case_sources(source_root: Union[str, Path]) -> List[Dict[str, Any]]:
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"FDG source directory not found: {root}")

    case_records: List[Dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if case_dir.name.startswith("."):
            continue
        volume_files = [path for path in case_dir.rglob("*") if path.is_file() and _is_volume_file(path)]
        if not volume_files:
            continue

        pet_candidates = [path for path in volume_files if any(keyword in path.name.lower() for keyword in ["pet", "suv"])]
        ct_candidates = [path for path in volume_files if "ct" in path.name.lower()]
        label_candidates = [
            path
            for path in volume_files
            if any(keyword in path.name.lower() for keyword in ["seg", "mask", "label", "lesion"])
        ]

        if len(volume_files) == 3 and not label_candidates:
            remaining = [path for path in volume_files if path not in pet_candidates and path not in ct_candidates]
            label_candidates = remaining

        pet_path = _pick_best_candidate(pet_candidates, ["pet", "suv"], "PET", case_dir)
        ct_path = _pick_best_candidate(ct_candidates, ["ct"], "CT", case_dir)
        label_path = _pick_best_candidate(label_candidates, ["seg", "mask", "label", "lesion"], "label", case_dir)

        case_id = _normalize_case_id(case_dir.name)
        metadata = _read_volume_metadata(label_path)
        case_records.append(
            {
                "case_id": case_id,
                "source_case_dir": str(case_dir),
                "source_pet": str(pet_path),
                "source_ct": str(ct_path),
                "source_label": str(label_path),
                **metadata,
            }
        )

    if not case_records:
        raise ValueError(f"No FDG PET/CT cases were discovered under {root}")
    return case_records


def _materialize_file(source: Path, destination: Path, link_mode: LinkMode) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    ensure_dir(destination.parent)
    if link_mode == "symlink":
        os.symlink(source, destination)
    elif link_mode == "copy":
        shutil.copy2(source, destination)
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported link mode: {link_mode}")


def normalize_fdg_cases(
    source_root: Union[str, Path],
    prepared_root: Union[str, Path],
    link_mode: LinkMode = "symlink",
) -> List[Dict[str, Any]]:
    prepared_root = ensure_dir(prepared_root)
    discovered_cases = discover_fdg_case_sources(source_root)
    normalized_cases: List[Dict[str, Any]] = []

    for case in discovered_cases:
        case_dir = ensure_dir(prepared_root / case["case_id"])
        pet_target = case_dir / "pet.nii.gz"
        ct_target = case_dir / "ct.nii.gz"
        label_target = case_dir / "label.nii.gz"
        _materialize_file(Path(case["source_pet"]), pet_target, link_mode)
        _materialize_file(Path(case["source_ct"]), ct_target, link_mode)
        _materialize_file(Path(case["source_label"]), label_target, link_mode)

        normalized_cases.append(
            {
                "case_id": case["case_id"],
                "case_dir": str(case_dir),
                "pet": str(pet_target),
                "ct": str(ct_target),
                "label": str(label_target),
                "lesion_positive": case["lesion_positive"],
                "positive_voxel_count": case["positive_voxel_count"],
                "shape_zyx": case["shape_zyx"],
                "spacing_xyz": case["spacing_xyz"],
            }
        )

    return normalized_cases


def build_fdg_manifest(
    source_root: Union[str, Path],
    prepared_root: Union[str, Path],
    manifest_path: Union[str, Path],
    seed: int = 42,
    link_mode: LinkMode = "symlink",
) -> Dict[str, Any]:
    cases = normalize_fdg_cases(source_root, prepared_root, link_mode=link_mode)
    manifest = {
        "dataset_name": "FDG-PET-CT-Lesions",
        "dataset_key": "autopet_fdg",
        "source_root": str(Path(source_root)),
        "prepared_root": str(Path(prepared_root)),
        "seed": seed,
        "case_count": len(cases),
        "positive_case_count": sum(1 for case in cases if case["lesion_positive"]),
        "negative_case_count": sum(1 for case in cases if not case["lesion_positive"]),
        "cases": cases,
    }
    save_json(manifest, manifest_path)
    return manifest


def load_fdg_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    return load_json(path)


def _sample_balanced_cases(cases: Sequence[Dict[str, Any]], target_count: int, rng_seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if target_count > len(cases):
        raise ValueError(f"Cannot sample {target_count} cases from a pool of {len(cases)}")

    positives = [case for case in cases if case["lesion_positive"]]
    negatives = [case for case in cases if not case["lesion_positive"]]
    if positives and negatives:
        positive_target = max(1, min(len(positives), round(target_count * len(positives) / len(cases))))
        negative_target = target_count - positive_target
        if negative_target == 0:
            negative_target = 1
            positive_target = target_count - 1
        if negative_target > len(negatives):
            negative_target = len(negatives)
            positive_target = target_count - negative_target
        if positive_target > len(positives):
            positive_target = len(positives)
            negative_target = target_count - positive_target
    else:
        positive_target = min(target_count, len(positives))
        negative_target = target_count - positive_target

    import random

    rng = random.Random(rng_seed)
    sampled_positives = rng.sample(positives, positive_target) if positive_target else []
    sampled_negatives = rng.sample(negatives, negative_target) if negative_target else []
    sampled_ids = {case["case_id"] for case in [*sampled_positives, *sampled_negatives]}
    sampled_cases = sorted([*sampled_positives, *sampled_negatives], key=lambda case: case["case_id"])
    remainder = [case for case in cases if case["case_id"] not in sampled_ids]
    return sampled_cases, remainder


def _build_split_record(split_name: str, train: Sequence[Dict[str, Any]], val: Sequence[Dict[str, Any]], review: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize(split_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        counts = Counter("positive" if case["lesion_positive"] else "negative" for case in split_cases)
        return {
            "case_ids": [case["case_id"] for case in split_cases],
            "positive": counts.get("positive", 0),
            "negative": counts.get("negative", 0),
            "count": len(split_cases),
        }

    return {
        "split_name": split_name,
        "train": summarize(train),
        "val": summarize(val),
        "review": summarize(review),
    }


def create_versioned_fdg_splits(
    manifest: Dict[str, Any],
    split_dir: Union[str, Path],
    seed: int = 42,
    dev_counts: Tuple[int, int, int] = (48, 8, 8),
) -> Dict[str, Dict[str, Any]]:
    cases = list(manifest["cases"])
    split_dir = ensure_dir(split_dir)

    dev_train_count, dev_val_count, dev_review_count = dev_counts
    total_dev = sum(dev_counts)
    if total_dev > len(cases):
        raise ValueError(f"fdg_dev requests {total_dev} cases but only {len(cases)} are available")

    dev_pool, remaining = _sample_balanced_cases(cases, total_dev, rng_seed=seed)
    dev_train, temp = _sample_balanced_cases(dev_pool, dev_train_count, rng_seed=seed + 1)
    dev_val, dev_review = _sample_balanced_cases(temp, dev_val_count, rng_seed=seed + 2)
    fdg_dev = _build_split_record("fdg_dev", dev_train, dev_val, dev_review)
    save_json(fdg_dev, Path(split_dir) / "fdg_dev.json")

    full_count = len(cases)
    full_train_count = max(1, int(round(full_count * 0.8)))
    full_val_count = max(1, int(round(full_count * 0.1)))
    full_review_count = full_count - full_train_count - full_val_count
    if full_review_count <= 0:
        full_review_count = 1
        full_train_count = max(1, full_train_count - 1)

    full_train, temp = _sample_balanced_cases(cases, full_train_count, rng_seed=seed + 10)
    full_val, full_review = _sample_balanced_cases(temp, full_val_count, rng_seed=seed + 11)
    fdg_full = _build_split_record("fdg_full", full_train, full_val, full_review)
    save_json(fdg_full, Path(split_dir) / "fdg_full.json")

    return {
        "fdg_dev": fdg_dev,
        "fdg_full": fdg_full,
    }


def _copy_or_link_volume(source: Union[str, Path], destination: Union[str, Path], link_mode: LinkMode) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    _materialize_file(source_path, destination_path, link_mode)


def resolve_dataset_name(dataset_id: int, suffix: str = "AutoPETFDG") -> str:
    return f"Dataset{dataset_id:03d}_{suffix}"


def export_nnunet_dataset(
    manifest: Dict[str, Any],
    split_record: Dict[str, Any],
    dataset_id: int,
    nnunet_raw_root: Union[str, Path],
    link_mode: LinkMode = "symlink",
) -> Dict[str, Any]:
    dataset_name = resolve_dataset_name(dataset_id)
    dataset_root = ensure_dir(Path(nnunet_raw_root) / dataset_name)
    images_tr = ensure_dir(dataset_root / "imagesTr")
    labels_tr = ensure_dir(dataset_root / "labelsTr")
    images_ts = ensure_dir(dataset_root / "imagesTs")
    labels_ts = ensure_dir(dataset_root / "labelsTs")

    case_lookup = {case["case_id"]: case for case in manifest["cases"]}
    ordered_train_ids = [*split_record["train"]["case_ids"], *split_record["val"]["case_ids"]]
    ordered_review_ids = list(split_record["review"]["case_ids"])

    mapping: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "split_name": split_record["split_name"],
        "train_case_ids": ordered_train_ids,
        "review_case_ids": ordered_review_ids,
        "cases": {},
    }

    for index, case_id in enumerate(ordered_train_ids):
        case = case_lookup[case_id]
        nnunet_case_id = f"FDG_{index:04d}"
        _copy_or_link_volume(case["pet"], images_tr / f"{nnunet_case_id}_0000.nii.gz", link_mode)
        _copy_or_link_volume(case["ct"], images_tr / f"{nnunet_case_id}_0001.nii.gz", link_mode)
        _copy_or_link_volume(case["label"], labels_tr / f"{nnunet_case_id}.nii.gz", link_mode)
        mapping["cases"][case_id] = {
            "nnunet_case_id": nnunet_case_id,
            "role": "train",
            "pet": case["pet"],
            "ct": case["ct"],
            "label": case["label"],
        }

    for review_offset, case_id in enumerate(ordered_review_ids, start=len(ordered_train_ids)):
        case = case_lookup[case_id]
        nnunet_case_id = f"FDG_{review_offset:04d}"
        _copy_or_link_volume(case["pet"], images_ts / f"{nnunet_case_id}_0000.nii.gz", link_mode)
        _copy_or_link_volume(case["ct"], images_ts / f"{nnunet_case_id}_0001.nii.gz", link_mode)
        _copy_or_link_volume(case["label"], labels_ts / f"{nnunet_case_id}.nii.gz", link_mode)
        mapping["cases"][case_id] = {
            "nnunet_case_id": nnunet_case_id,
            "role": "review",
            "pet": case["pet"],
            "ct": case["ct"],
            "label": case["label"],
        }

    dataset_json = {
        "channel_names": {"0": "PET_SUV", "1": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(ordered_train_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    save_json(dataset_json, dataset_root / "dataset.json")
    save_json(mapping, dataset_root / "case_mapping.json")
    return mapping

