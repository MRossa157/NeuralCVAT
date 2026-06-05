from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural_cvat.types import COCO_CATEGORIES


def _default_coco_skeleton() -> dict[str, Any]:
    return {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }


def validate_coco(data: dict[str, Any]) -> None:
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"COCO missing required key: {key}")
    image_ids = {img["id"] for img in data["images"]}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            raise ValueError(
                f"Annotation {ann.get('id')} references unknown image_id {ann['image_id']}",
            )


def load_coco(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    validate_coco(data)
    return data


def save_coco(data: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "categories" not in data:
        data["categories"] = COCO_CATEGORIES
    validate_coco(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def find_images_dir(coco_root: Path) -> Path:
    for candidate in (
        coco_root / "images" / "default",
        coco_root / "images",
        coco_root,
    ):
        if candidate.is_dir() and (any(candidate.glob("*.jpg")) or any(candidate.glob("*.png"))):
            return candidate
        if candidate.is_dir():
            for sub in candidate.iterdir():
                if sub.is_dir() and (any(sub.glob("*.jpg")) or any(sub.glob("*.png"))):
                    return sub
    raise FileNotFoundError(f"No images directory found under {coco_root}")


def default_annotation_path(coco_root: Path) -> Path:
    ann = coco_root / "annotations" / "instances_default.json"
    if ann.exists():
        return ann
    for p in coco_root.rglob("instances_*.json"):
        return p
    raise FileNotFoundError(f"No COCO annotation json under {coco_root}")
