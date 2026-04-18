"""Brain tumor classification + XAI baseline."""

from __future__ import annotations

from typing import Any

__all__ = ["build_resnet18_binary"]


def __getattr__(name: str) -> Any:
    if name == "build_resnet18_binary":
        # Lazy import keeps lightweight utility scripts usable even when
        # heavy ML dependencies are not installed in the current environment.
        from .model import build_resnet18_binary as _build_resnet18_binary

        return _build_resnet18_binary
    raise AttributeError(f"module 'brain_tumor_xai' has no attribute {name!r}")
