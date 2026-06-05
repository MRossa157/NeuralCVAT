from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from neural_cvat.dataset.coco_io import load_coco, save_coco

# Максимум verified-кадров (с хотя бы одной аннотацией) в обучающем датасете
VERIFIED_IMAGE_LIMIT = 5


def _reindex_coco(data: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    images = [img for img in data["images"] if img["id"] in image_ids]
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]

    old_to_new: dict[int, int] = {}
    for new_id, img in enumerate(images, start=1):
        old_to_new[img["id"]] = new_id
        img["id"] = new_id

    for new_id, ann in enumerate(annotations, start=1):
        ann["id"] = new_id
        ann["image_id"] = old_to_new[ann["image_id"]]

    data["images"] = images
    data["annotations"] = annotations
    return data


def trim_to_limit(
    coco: dict[str, Any] | Path,
    out_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Verified-кадры (с аннотациями), не более VERIFIED_IMAGE_LIMIT штук."""
    if isinstance(coco, Path):
        data = load_coco(coco)
    else:
        data = dict(coco)
        data["images"] = list(data["images"])
        data["annotations"] = list(data["annotations"])

    annotated_image_ids = {ann["image_id"] for ann in data["annotations"]}
    data = _reindex_coco(data, annotated_image_ids)

    if len(data["images"]) > VERIFIED_IMAGE_LIMIT:
        rng = random.Random(seed)
        chosen = rng.sample(data["images"], VERIFIED_IMAGE_LIMIT)
        keep_ids = {img["id"] for img in chosen}
        data = _reindex_coco(data, keep_ids)

    if out_path is not None:
        save_coco(data, out_path)
    return data


def trim_to_verified(
    coco: dict[str, Any] | Path,
    out_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    return trim_to_limit(coco, out_path=out_path, seed=seed)
