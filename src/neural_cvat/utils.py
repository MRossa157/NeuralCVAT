from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_yaml_config(path: Path | str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
