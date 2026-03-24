from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: Dict[str, Any], path: Union[str, Path]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device: Optional[str] = None) -> torch.device:
    if device:
        target = torch.device(device)
        if target.type == "cpu":
            torch.backends.mkldnn.enabled = False
        return target
    if torch.cuda.is_available():
        return torch.device("cuda")
    torch.backends.mkldnn.enabled = False
    return torch.device("cpu")
