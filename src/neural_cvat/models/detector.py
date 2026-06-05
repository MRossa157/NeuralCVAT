from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from neural_cvat.types import Detection


class Detector(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name: str, num_classes: int = 4, **kwargs: Any) -> Detector: ...

    @abstractmethod
    def fit(
        self,
        coco_train: Path,
        coco_val: Path,
        images_dir: Path,
        output_dir: Path,
        **cfg: Any,
    ) -> Path: ...

    @abstractmethod
    def predict(
        self,
        image_paths: list[Path],
        tile_for_plate: bool = False,
        **kwargs: Any,
    ) -> list[list[Detection]]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs: Any) -> Detector: ...


def build_detector(model_name: str, num_classes: int = 4, **kwargs: Any) -> Detector:
    name = model_name.lower()
    if name.endswith(".pt") or name.startswith("yolo"):
        from neural_cvat.models.yolo_detector import YoloDetector

        return YoloDetector.from_pretrained(model_name, num_classes=num_classes, **kwargs)
    from neural_cvat.models.hf_detector import HfDetector

    return HfDetector.from_pretrained(model_name, num_classes=num_classes, **kwargs)
