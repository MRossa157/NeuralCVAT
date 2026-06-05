from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_cvat.dataset.crops import extract_crops_by_class
from neural_cvat.models.classifier import AttributeClassifier


def train_classifier(
    coco_path: Path | str,
    images_dir: Path | str,
    class_name: str,
    output_path: Path | str,
    crops_dir: Path | str | None = None,
    **cfg: Any,
) -> Path:
    coco_path = Path(coco_path)
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    crops_root = Path(crops_dir) if crops_dir else output_path.parent / "crops"
    crops_path = extract_crops_by_class(coco_path, images_dir, crops_root, class_name)

    clf = AttributeClassifier.create(
        class_name,
        backbone=cfg.get("backbone", "efficientnet_b0"),
        image_size=int(cfg.get("image_size", 224)),
    )
    return clf.fit(
        crops_path,
        output_path,
        epochs=int(cfg.get("epochs", 20)),
        batch_size=int(cfg.get("batch", 32)),
        lr=float(cfg.get("lr", 1e-3)),
    )
