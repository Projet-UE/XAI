from __future__ import annotations

import numpy as np

from autopet_xai.metrics import compute_case_metrics


def test_segmentation_metrics_include_dice_and_false_volumes() -> None:
    ground_truth = np.zeros((4, 8, 8), dtype=np.uint8)
    prediction = np.zeros_like(ground_truth)
    ground_truth[:, 2:4, 2:4] = 1
    prediction[:, 3:5, 2:4] = 1

    metrics = compute_case_metrics(ground_truth, prediction, spacing_xyz=(2.0, 2.0, 2.0))
    assert 0.0 <= metrics["dice"] <= 1.0
    assert metrics["false_negative_voxels"] > 0
    assert metrics["false_positive_voxels"] > 0
    assert metrics["false_negative_volume_ml"] > 0
    assert metrics["false_positive_volume_ml"] > 0
