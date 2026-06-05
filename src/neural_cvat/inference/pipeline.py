from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from neural_cvat.dataset.coco_export import predictions_to_coco10, read_image_size
from neural_cvat.dataset.crops import crop_bbox
from neural_cvat.models.classifier import AttributeClassifier
from neural_cvat.models.detector import Detector
from neural_cvat.models.plate_ocr import PlateOCR
from neural_cvat.types import CATEGORY_BAG, CATEGORY_BIKE, CATEGORY_PLATE, Detection


class InferencePipeline:
    def __init__(
        self,
        detector: Detector,
        plate_ocr: PlateOCR | None = None,
        bag_clf: AttributeClassifier | None = None,
        bike_clf: AttributeClassifier | None = None,
        tile_for_plate: bool = False,
    ) -> None:
        self.detector = detector
        self.plate_ocr = plate_ocr or PlateOCR()
        self.bag_clf = bag_clf
        self.bike_clf = bike_clf
        self.tile_for_plate = tile_for_plate

    def run(
        self,
        image_paths: list[Path | str],
        out_path: Path | str | None = None,
    ) -> dict[str, Any]:
        paths = [Path(p) for p in image_paths]
        all_predictions: list[tuple[Path, int, int, list[Detection]]] = []

        batch_dets = self.detector.predict(paths, tile_for_plate=self.tile_for_plate)
        for path, dets in tqdm(
            zip(paths, batch_dets, strict=True),
            total=len(paths),
            desc="infer",
            unit="img",
        ):
            image = cv2.imread(str(path))
            if image is None:
                continue
            enriched = self._enrich_detections(image, dets)
            w, h = read_image_size(path)
            all_predictions.append((path, w, h, enriched))

        return predictions_to_coco10(all_predictions, out_path)

    def _enrich_detections(self, image, dets: list[Detection]) -> list[Detection]:
        result: list[Detection] = []
        for det in dets:
            crop = crop_bbox(image, list(det.bbox_xywh))
            attrs = dict(det.attributes)
            if det.category_id == CATEGORY_PLATE:
                attrs["plate_number"] = self.plate_ocr.read(crop)
            elif det.category_id == CATEGORY_BAG and self.bag_clf is not None:
                attrs.update(self.bag_clf.predict(crop))
            elif det.category_id == CATEGORY_BIKE and self.bike_clf is not None:
                attrs.update(self.bike_clf.predict(crop))
            result.append(
                Detection(
                    bbox_xywh=det.bbox_xywh,
                    category_id=det.category_id,
                    score=det.score,
                    attributes=attrs,
                ),
            )
        return result
