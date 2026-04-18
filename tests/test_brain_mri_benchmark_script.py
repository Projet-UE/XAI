from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


brain_benchmark = _load_script_module("brain_mri_benchmark_xai_methods.py")


def test_top_fraction_mask_selects_expected_pixel_count() -> None:
    heatmap = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = brain_benchmark._top_fraction_mask(heatmap, 0.2)
    assert mask.shape == (10, 10)
    assert int(mask.sum()) == 20


def test_apply_mask_zeroes_masked_positions() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    masked = brain_benchmark._apply_mask(image, mask, baseline=0.0)
    assert torch.all(masked[:, :2, :2] == 0.0)
    assert torch.all(masked[:, 2:, 2:] == 1.0)


def test_bootstrap_ci_is_well_formed() -> None:
    ci = brain_benchmark._bootstrap_ci([0.1, 0.2, 0.3, 0.4], iterations=500, seed=7)
    assert ci is not None
    assert ci["low"] <= ci["mean"] <= ci["high"]
    assert ci["sample_count"] == 4.0
