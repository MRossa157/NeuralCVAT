from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_cvat.dataset.yolo_export import coco_to_yolo
from neural_cvat.models.detector import Detector, build_detector


def train_detector(
    coco_train: Path | str,
    coco_val: Path | str,
    images_dir: Path | str,
    model_name: str,
    output_dir: Path | str,
    **cfg: Any,
) -> Path:
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)
    detector: Detector = build_detector(model_name, **cfg)
    print(f"Training detector: {model_name} (Ultralytics shows epoch progress below)")

    if model_name.lower().endswith(".pt") or model_name.lower().startswith("yolo"):
        yolo_dir = output_dir / "yolo_dataset"
        coco_to_yolo(
            coco_train,
            images_dir,
            yolo_dir,
            splits={"train": coco_train, "val": coco_val},
        )

    return detector.fit(
        Path(coco_train),
        Path(coco_val),
        images_dir,
        output_dir,
        **cfg,
    )
