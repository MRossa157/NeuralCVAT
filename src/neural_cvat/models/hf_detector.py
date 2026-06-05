from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from neural_cvat.dataset.coco_io import load_coco
from neural_cvat.models.detector import Detector
from neural_cvat.types import CATEGORY_PLATE, Detection
from neural_cvat.utils import resolve_torch_device


class CocoTorchDataset(Dataset):
    def __init__(self, coco_path: Path, images_dir: Path, processor) -> None:
        self.data = load_coco(coco_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.images = {img["id"]: img for img in self.data["images"]}
        self.annotations = self.data["annotations"]
        self.image_ids = sorted(self.images.keys())
        self.id_to_label = {cat["id"]: idx for idx, cat in enumerate(self.data["categories"])}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        img_info = self.images[image_id]
        path = self.images_dir / img_info["file_name"]
        if not path.exists():
            for alt in self.images_dir.rglob(img_info["file_name"]):
                path = alt
                break
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = [a for a in self.annotations if a["image_id"] == image_id]
        coco_anns = []
        for ann in anns:
            coco_anns.append(
                {
                    "bbox": ann["bbox"],
                    "category_id": self.id_to_label[ann["category_id"]],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                },
            )

        encoding = self.processor(images=image, annotations=coco_anns, return_tensors="pt")
        if "labels" in encoding:
            encoding["class_labels"] = encoding.pop("labels")
        return {k: v[0] if hasattr(v, "shape") and v.ndim > 0 else v for k, v in encoding.items()}


class HfDetector(Detector):
    def __init__(
        self,
        model,
        processor,
        label_to_id: dict[int, int],
        conf: float = 0.25,
        plate_conf: float = 0.15,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._processor = processor
        self._label_to_id = label_to_id
        self.conf = conf
        self.plate_conf = plate_conf
        self.device = device or resolve_torch_device()
        self._model.to(self.device)

    @classmethod
    def from_pretrained(cls, model_name: str, num_classes: int = 4, **kwargs: Any) -> HfDetector:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        label_to_id = {i: i + 1 for i in range(num_classes)}
        if hasattr(model.config, "label2id") and model.config.label2id:
            label_to_id = {int(k): int(v) for k, v in model.config.label2id.items()}
        return cls(
            model,
            processor,
            label_to_id,
            conf=kwargs.get("conf", 0.25),
            plate_conf=kwargs.get("plate_conf", 0.15),
        )

    def fit(
        self,
        coco_train: Path,
        coco_val: Path,
        images_dir: Path,
        output_dir: Path,
        **cfg: Any,
    ) -> Path:
        from transformers import Trainer, TrainingArguments

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = CocoTorchDataset(Path(coco_train), Path(images_dir), self._processor)
        val_ds = CocoTorchDataset(Path(coco_val), Path(images_dir), self._processor)

        args = TrainingArguments(
            output_dir=str(output_dir / "hf_detector"),
            per_device_train_batch_size=int(cfg.get("batch", 4)),
            num_train_epochs=int(cfg.get("epochs", 30)),
            learning_rate=float(cfg.get("lr", 5e-5)),
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        def collate_fn(batch):
            return self._processor.pad(batch, return_tensors="pt")

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
        )
        trainer.train()
        save_path = output_dir / "hf_weights"
        self.save(save_path)
        return save_path

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._processor.save_pretrained(path)
        meta = {"label_to_id": self._label_to_id, "conf": self.conf, "plate_conf": self.plate_conf}
        (path / "neural_cvat_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> HfDetector:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        path = Path(path)
        processor = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForObjectDetection.from_pretrained(path)
        meta_path = path / "neural_cvat_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            model,
            processor,
            meta.get("label_to_id", {i: i + 1 for i in range(4)}),
            conf=meta.get("conf", kwargs.get("conf", 0.25)),
            plate_conf=meta.get("plate_conf", kwargs.get("plate_conf", 0.15)),
        )

    def predict(
        self,
        image_paths: list[Path],
        tile_for_plate: bool = False,
        **kwargs: Any,
    ) -> list[list[Detection]]:
        del tile_for_plate
        out: list[list[Detection]] = []
        for path in image_paths:
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([[image.shape[0], image.shape[1]]])
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=min(self.conf, self.plate_conf),
                target_sizes=target_sizes,
            )[0]
            dets = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"],
                strict=True,
            ):
                cls_id = self._label_to_id.get(int(label.item()), int(label.item()) + 1)
                conf = float(score.item())
                threshold = self.plate_conf if cls_id == CATEGORY_PLATE else self.conf
                if conf < threshold:
                    continue
                xyxy = box.tolist()
                dets.append(
                    Detection(
                        bbox_xywh=(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
                        category_id=cls_id,
                        score=conf,
                    ),
                )
            out.append(dets)
        return out
