from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:] = rng.integers(0, 40, size=(1, 1, 3))
    center = 16 + (seed % 24)
    radius = 8 + (seed % 6)
    yy, xx = np.ogrid[:64, :64]
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= radius ** 2
    image[mask] = rng.integers(180, 255, size=(1, 1, 3))
    return image


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "brain-mri-images"
    for class_name, offset in [("no", 0), ("yes", 100)]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            image = _make_image(seed=offset + index)
            Image.fromarray(image).save(class_dir / f"{class_name}_{index}.png")
    return root
