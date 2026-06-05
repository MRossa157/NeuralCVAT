from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from neural_cvat.dataset.coco_io import load_coco
from neural_cvat.types import (
    BAG_ATTRIBUTES,
    BIKE_ATTRIBUTES,
    CATEGORY_BAG,
    CATEGORY_BIKE,
)


def crop_bbox(image, bbox_xywh: list[float], pad: float = 0.02):
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh
    pad_x = bw * pad
    pad_y = bh * pad
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(w, int(x + bw + pad_x))
    y2 = min(h, int(y + bh + pad_y))
    return image[y1:y2, x1:x2]


def extract_crops_by_class(
    coco_path: Path | str,
    images_dir: Path | str,
    out_dir: Path | str,
    class_name: str,
) -> Path:
    coco_path = Path(coco_path)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco(coco_path)
    category_id = {"bag": CATEGORY_BAG, "bike": CATEGORY_BIKE}.get(class_name)
    if category_id is None:
        raise ValueError(f"extract_crops_by_class supports bag/bike, got {class_name}")

    attr_names = list(BAG_ATTRIBUTES if class_name == "bag" else BIKE_ATTRIBUTES)
    meta: list[dict[str, Any]] = []

    images_by_id = {img["id"]: img for img in data["images"]}
    target_anns = [a for a in data["annotations"] if a["category_id"] == category_id]
    index = 0
    for ann in tqdm(target_anns, desc=f"crops {class_name}", unit="ann"):
        img_info = images_by_id.get(ann["image_id"])
        if not img_info:
            continue
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            for alt in images_dir.rglob(img_info["file_name"]):
                img_path = alt
                break
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        crop = crop_bbox(image, ann["bbox"])
        if crop.size == 0:
            continue

        attrs = ann.get("attributes", {})
        fname = f"{index:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), crop)
        entry = {"file": fname, "annotation_id": ann["id"], "image_id": ann["image_id"]}
        for name in attr_names:
            entry[name] = bool(attrs.get(name, False))
        meta.append(entry)
        index += 1

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
