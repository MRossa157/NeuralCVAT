from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: Path | str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
