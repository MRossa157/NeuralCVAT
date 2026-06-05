from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from neural_cvat.dataset.coco_io import load_coco, save_coco


def _image_label_vector(data: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    cats_per_image: dict[int, set[int]] = defaultdict(set)
    for ann in data["annotations"]:
        cats_per_image[ann["image_id"]].add(ann["category_id"])
    return {iid: tuple(sorted(cats)) for iid, cats in cats_per_image.items()}


def split_train_val(
    coco_path: Path | str,
    out_dir: Path | str,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path]:
    coco_path = Path(coco_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco(coco_path)
    image_ids = [img["id"] for img in data["images"]]
    labels = [_image_label_vector(data).get(iid, ()) for iid in image_ids]

    try:
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels if len(set(map(str, labels))) > 1 else None,
        )
    except ValueError:
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=val_ratio,
            random_state=seed,
        )

    train_set, val_set = set(train_ids), set(val_ids)

    def subset(ids: set[int]) -> dict[str, Any]:
        images = [img for img in data["images"] if img["id"] in ids]
        annotations = [ann for ann in data["annotations"] if ann["image_id"] in ids]
        return {
            **{k: v for k, v in data.items() if k not in ("images", "annotations")},
            "images": images,
            "annotations": annotations,
        }

    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"
    save_coco(subset(train_set), train_path)
    save_coco(subset(val_set), val_path)
    return train_path, val_path
