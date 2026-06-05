from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from neural_cvat.dataset.yolo_export import coco_to_yolo
from neural_cvat.models.detector import Detector
from neural_cvat.types import CATEGORY_PLATE, Detection
from neural_cvat.utils import resolve_torch_device


class YoloDetector(Detector):
    def __init__(
        self,
        model,
        conf: float = 0.25,
        iou: float = 0.45,
        plate_conf: float = 0.15,
        imgsz: int = 1280,
    ) -> None:
        self._model = model
        self.conf = conf
        self.iou = iou
        self.plate_conf = plate_conf
        self.imgsz = imgsz
        self._weights_path: Path | None = None

    @classmethod
    def from_pretrained(cls, model_name: str, num_classes: int = 4, **kwargs: Any) -> YoloDetector:
        from ultralytics import YOLO

        model = YOLO(model_name)
        return cls(
            model,
            conf=kwargs.get("conf", 0.25),
            iou=kwargs.get("iou", 0.45),
            plate_conf=kwargs.get("plate_conf", 0.15),
            imgsz=kwargs.get("imgsz", 1280),
        )

    def fit(
        self,
        coco_train: Path,
        coco_val: Path,
        images_dir: Path,
        output_dir: Path,
        **cfg: Any,
    ) -> Path:
        output_dir = Path(output_dir)
        yolo_dir = output_dir / "yolo_dataset"
        data_yaml = coco_to_yolo(
            coco_train,
            images_dir,
            yolo_dir,
            splits={"train": coco_train, "val": coco_val},
        )

        epochs = int(cfg.get("epochs", 50))
        batch = int(cfg.get("batch", 8))
        imgsz = int(cfg.get("imgsz", self.imgsz))

        results = self._model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=resolve_torch_device(),
            project=str(output_dir),
            name="detector",
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.3,
            exist_ok=True,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        self._model = type(self._model)(str(best))
        self._weights_path = best
        return best

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._weights_path and self._weights_path.exists():
            shutil.copy2(self._weights_path, path)
        else:
            self._model.save(str(path))

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> YoloDetector:
        det = cls.from_pretrained(str(path), **kwargs)
        det._weights_path = Path(path)
        return det

    def predict(
        self,
        image_paths: list[Path],
        tile_for_plate: bool = False,
        **kwargs: Any,
    ) -> list[list[Detection]]:
        if tile_for_plate:
            return [self._predict_tiled(p) for p in image_paths]
        return [self._predict_single(p) for p in image_paths]

    def _predict_single(self, image_path: Path) -> list[Detection]:
        results = self._model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=min(self.conf, self.plate_conf),
            iou=self.iou,
            device=resolve_torch_device(),
            verbose=False,
        )
        return self._parse_results(results[0])

    def _predict_tiled(self, image_path: Path) -> list[Detection]:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            return self._predict_single(image_path)

        weights = self._weights_path or getattr(self._model, "ckpt_path", None) or "yolo11n.pt"
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(weights),
            confidence_threshold=self.plate_conf,
            device=resolve_torch_device(),
        )
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        dets: list[Detection] = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            dets.append(
                Detection(
                    bbox_xywh=(bbox.minx, bbox.miny, bbox.maxx - bbox.minx, bbox.maxy - bbox.miny),
                    category_id=int(obj.category.id) + 1,
                    score=float(obj.score.value),
                ),
            )
        return self._filter_by_class_conf(dets)

    def _parse_results(self, result) -> list[Detection]:
        dets: list[Detection] = []
        if result.boxes is None:
            return dets
        for box in result.boxes:
            cls_id = int(box.cls.item()) + 1
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            dets.append(
                Detection(
                    bbox_xywh=(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
                    category_id=cls_id,
                    score=conf,
                ),
            )
        return self._filter_by_class_conf(dets)

    def _filter_by_class_conf(self, dets: list[Detection]) -> list[Detection]:
        filtered = []
        for d in dets:
            threshold = self.plate_conf if d.category_id == CATEGORY_PLATE else self.conf
            if d.score >= threshold:
                filtered.append(d)
        return filtered
