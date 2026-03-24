from __future__ import annotations

from typing import Any, Dict, Optional, Union

import warnings

import torch
from torch import nn
from torchvision import models


def build_resnet18_binary(pretrained: bool = True) -> nn.Module:
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT

    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover - depends on remote weight availability
        warnings.warn(
            f"Failed to load pretrained ResNet18 weights ({exc}). Falling back to randomly initialized weights.",
            stacklevel=2,
        )
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def save_checkpoint(model: nn.Module, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, checkpoint_path: str, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return payload
