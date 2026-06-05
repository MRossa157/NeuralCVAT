from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from neural_cvat.dataset.coco_io import (
    default_annotation_path,
    find_images_dir,
    load_coco,
    save_coco,
)
from neural_cvat.dataset.trim import trim_to_limit
from neural_cvat.types import COCO_CATEGORIES


def _resolve_coco_dir(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    if (path / "annotations").exists() or list(path.rglob("instances_*.json")):
        ann = default_annotation_path(path)
        images = find_images_dir(path)
        return ann, images
    if path.suffix == ".json":
        return path, find_images_dir(path.parent)
    raise FileNotFoundError(f"Cannot resolve COCO layout in {path}")


def merge_coco_dirs(
    input_dirs: list[Path | str],
    out_dir: Path | str,
    trim: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_images = out_dir / "images" / "default"
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann = out_dir / "annotations" / "instances_default.json"

    merged: dict[str, Any] = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"description": "merged neural_cvat dataset"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    next_image_id = 1
    next_ann_id = 1
    seen_files: set[str] = set()

    for raw_dir in input_dirs:
        ann_path, images_dir = _resolve_coco_dir(Path(raw_dir))
        data = load_coco(ann_path)
        if trim:
            data = trim_to_limit(data)

        old_to_new: dict[int, int] = {}
        for img in data["images"]:
            fname = img["file_name"]
            if fname in seen_files:
                continue
            seen_files.add(fname)
            src = images_dir / fname
            if not src.exists():
                for alt in images_dir.rglob(fname):
                    src = alt
                    break
            if src.exists():
                shutil.copy2(src, out_images / fname)
            old_to_new[img["id"]] = next_image_id
            merged["images"].append(
                {
                    "id": next_image_id,
                    "file_name": fname,
                    "width": img["width"],
                    "height": img["height"],
                    "license": img.get("license", 0),
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0,
                },
            )
            next_image_id += 1

        for ann in data["annotations"]:
            new_image_id = old_to_new.get(ann["image_id"])
            if new_image_id is None:
                continue
            merged["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": new_image_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann.get("segmentation", []),
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann.get("iscrowd", 0),
                    "attributes": ann.get("attributes", {}),
                },
            )
            next_ann_id += 1

    save_coco(merged, out_ann)
    return out_ann
