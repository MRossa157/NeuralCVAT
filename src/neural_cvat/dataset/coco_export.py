from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from neural_cvat.dataset.coco_io import save_coco
from neural_cvat.models.plate_ocr import extract_plate_number
from neural_cvat.types import (
    CATEGORY_BAG,
    CATEGORY_BIKE,
    CATEGORY_CAR,
    CATEGORY_PLATE,
    COCO_CATEGORIES,
    Detection,
)


def _default_attributes(category_id: int, attrs: dict[str, Any]) -> dict[str, Any]:
    base = {"occluded": False, "rotation": 0}
    if category_id == CATEGORY_PLATE:
        plate = attrs.get("plate_number", "")
        if plate:
            plate = extract_plate_number(str(plate)) or str(plate)
        return {**base, "plate_number": plate}
    if category_id == CATEGORY_CAR:
        return base
    if category_id == CATEGORY_BAG:
        return {
            **base,
            "is_clean": bool(attrs.get("is_clean", False)),
            "is_deptrans_format": bool(attrs.get("is_deptrans_format", False)),
            "is_defected": bool(attrs.get("is_defected", False)),
        }
    if category_id == CATEGORY_BIKE:
        return {**base, "is_good_appearance": bool(attrs.get("is_good_appearance", False))}
    return base


def predictions_to_coco10(
    predictions: list[tuple[Path, int, int, list[Detection]]],
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    coco: dict[str, Any] = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"description": "neural_cvat predictions"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    ann_id = 1
    for image_path, width, height, dets in predictions:
        coco["images"].append(
            {
                "id": image_id,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            },
        )
        for det in dets:
            x, y, w, h = det.bbox_xywh
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det.category_id,
                    "segmentation": [],
                    "area": float(w * h),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0,
                    "attributes": _default_attributes(det.category_id, det.attributes),
                },
            )
            ann_id += 1
        image_id += 1

    if out_path is not None:
        save_coco(coco, out_path)
    return coco


def read_image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return w, h
