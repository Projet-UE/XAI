from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from brain_tumor_xai.utils import ensure_dir, save_json

from .data import resolve_dataset_name


def build_nnunet_environment(artifacts_dir: Union[str, Path]) -> Dict[str, str]:
    artifacts_dir = Path(artifacts_dir)
    nnunet_raw = ensure_dir(artifacts_dir / "nnunet_raw")
    nnunet_preprocessed = ensure_dir(artifacts_dir / "nnunet_preprocessed")
    nnunet_results = ensure_dir(artifacts_dir / "nnunet_results")
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(nnunet_raw),
            "nnUNet_preprocessed": str(nnunet_preprocessed),
            "nnUNet_results": str(nnunet_results),
        }
    )
    save_json(
        {
            "nnUNet_raw": str(nnunet_raw),
            "nnUNet_preprocessed": str(nnunet_preprocessed),
            "nnUNet_results": str(nnunet_results),
        },
        artifacts_dir / "nnunet_environment.json",
    )
    return environment


def run_command(command: Sequence[str], env: Dict[str, str], cwd: Optional[Union[str, Path]] = None) -> None:
    subprocess.run(list(command), check=True, env=env, cwd=str(cwd) if cwd else None)


def patch_nnunet_torch_compatibility() -> Optional[Path]:
    try:
        from nnunetv2.training.lr_scheduler import polylr
    except ImportError:
        return None

    module_path = Path(inspect.getsourcefile(polylr) or inspect.getfile(polylr))
    source = module_path.read_text(encoding="utf-8")
    legacy_call = "super().__init__(optimizer, current_step if current_step is not None else -1, False)"
    compatible_call = "super().__init__(optimizer, current_step if current_step is not None else -1)"
    if legacy_call not in source:
        return None

    module_path.write_text(source.replace(legacy_call, compatible_call), encoding="utf-8")
    return module_path


def plan_and_preprocess(
    dataset_id: int,
    artifacts_dir: Union[str, Path],
    verify_dataset_integrity: bool = True,
) -> Dict[str, str]:
    env = build_nnunet_environment(artifacts_dir)
    command = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id)]
    if verify_dataset_integrity:
        command.append("--verify_dataset_integrity")
    run_command(command, env)
    return env


def preprocessed_dataset_exists(dataset_id: int, artifacts_dir: Union[str, Path]) -> bool:
    dataset_name = resolve_dataset_name(dataset_id)
    preprocessed_root = Path(artifacts_dir) / "nnunet_preprocessed" / dataset_name
    return (preprocessed_root / "nnUNetPlans.json").exists()


def train_model(
    dataset_id: int,
    configuration: str,
    fold: Union[int, str],
    artifacts_dir: Union[str, Path],
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    device: Optional[str] = None,
) -> Dict[str, str]:
    patch_nnunet_torch_compatibility()
    env = build_nnunet_environment(artifacts_dir)
    command = ["nnUNetv2_train", str(dataset_id), configuration, str(fold), "-tr", trainer, "-p", plans]
    if device:
        command.extend(["-device", device])
    run_command(command, env)
    return env


def predict_cases(
    dataset_id: int,
    configuration: str,
    fold: Union[int, str],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    artifacts_dir: Union[str, Path],
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    checkpoint_name: str = "checkpoint_best.pth",
    device: Optional[str] = None,
    save_probabilities: bool = True,
) -> Dict[str, str]:
    env = build_nnunet_environment(artifacts_dir)
    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-f",
        str(fold),
        "-tr",
        trainer,
        "-p",
        plans,
        "-chk",
        checkpoint_name,
    ]
    if device:
        command.extend(["-device", device])
    if save_probabilities:
        command.append("--save_probabilities")
    run_command(command, env)
    return env


def resolve_training_output_dir(
    dataset_id: int,
    configuration: str,
    artifacts_dir: Union[str, Path],
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
) -> Path:
    dataset_name = resolve_dataset_name(dataset_id)
    return Path(artifacts_dir) / "nnunet_results" / dataset_name / f"{trainer}__{plans}__{configuration}"
