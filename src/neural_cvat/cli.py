from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
import yaml

from neural_cvat.dataset.coco_io import (
    default_annotation_path,
    find_images_dir,
    load_coco,
)
from neural_cvat.dataset.merge import merge_coco_dirs
from neural_cvat.dataset.split import split_train_val
from neural_cvat.dataset.trim import VERIFIED_IMAGE_LIMIT, trim_to_limit
from neural_cvat.inference.pipeline import InferencePipeline
from neural_cvat.models.classifier import AttributeClassifier
from neural_cvat.models.detector import build_detector
from neural_cvat.models.plate_ocr import PlateOCR
from neural_cvat.train.classifier import train_classifier
from neural_cvat.train.detector import train_detector

app = typer.Typer(help="NeuralCVAT: dataset prep, training, and CVAT pre-annotation")


def _load_config(config: Path | None) -> dict:
    if config is None:
        return {}
    with config.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.command("trim-dataset")
def trim_dataset(
    backup: Path = typer.Argument(
        ...,
        help="Папка backup_coco_* или путь к instances_default.json",
    ),
    out: Path = typer.Option(..., "--out", help="Куда сохранить обрезанный COCO"),
    seed: int = typer.Option(42, "--seed"),
    val_ratio: float | None = typer.Option(
        None,
        "--val-ratio",
        help="Если задан — train/val split",
    ),
) -> None:
    backup = Path(backup)
    ann = backup if backup.suffix == ".json" else default_annotation_path(backup)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    out_ann = out / "annotations" / "instances_default.json"

    data = trim_to_limit(ann, out_ann, seed=seed)

    src_images = find_images_dir(backup if backup.suffix != ".json" else backup.parent.parent)
    dst_images = out / "images" / "default"
    dst_images.mkdir(parents=True, exist_ok=True)
    for img in data["images"]:
        src = src_images / img["file_name"]
        if not src.exists():
            for alt in src_images.rglob(img["file_name"]):
                src = alt
                break
        if src.exists():
            shutil.copy2(src, dst_images / img["file_name"])

    typer.echo(
        f"Images: {len(data['images'])} (limit {VERIFIED_IMAGE_LIMIT}), "
        f"annotations: {len(data['annotations'])}",
    )
    typer.echo(f"Saved: {out_ann}")

    if val_ratio is not None:
        split_train_val(out_ann, out / "splits", val_ratio=val_ratio, seed=seed)
        typer.echo(f"Splits: {out / 'splits'}")


@app.command("prepare-dataset")
def prepare_dataset(
    inputs: list[Path] = typer.Argument(..., help="Один или несколько backup_coco_*"),
    out: Path = typer.Option(..., "--out", help="Merged dataset directory"),
    val_ratio: float = typer.Option(0.15, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    if len(inputs) == 1:
        backup = Path(inputs[0])
        ann = default_annotation_path(backup)
        out_ann = out / "annotations" / "instances_default.json"
        trim_to_limit(ann, out_ann, seed=seed)
        src_images = find_images_dir(backup)
        dst = out / "images" / "default"
        dst.mkdir(parents=True, exist_ok=True)
        for img in load_coco(out_ann)["images"]:
            src = src_images / img["file_name"]
            if src.exists():
                shutil.copy2(src, dst / img["file_name"])
    else:
        merge_coco_dirs(inputs, out, trim=True)

    ann = out / "annotations" / "instances_default.json"
    split_train_val(ann, out / "splits", val_ratio=val_ratio, seed=seed)
    typer.echo(f"Dataset: {out}")
    typer.echo(f"Train/val splits: {out / 'splits'}")


@app.command("train-detector")
def cmd_train_detector(
    coco_train: Path = typer.Option(..., "--coco-train"),
    coco_val: Path = typer.Option(..., "--coco-val"),
    images: Path = typer.Option(..., "--images"),
    model_name: str = typer.Option("yolo11n.pt", "--model-name"),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    batch: Optional[int] = typer.Option(None, "--batch"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz"),
) -> None:
    cfg = _load_config(config)
    train_cfg = {
        "epochs": epochs or cfg.get("epochs", 50),
        "batch": batch or cfg.get("batch", 8),
        "imgsz": imgsz or cfg.get("imgsz", 1280),
        **cfg.get("detector", {}),
    }
    weights = train_detector(coco_train, coco_val, images, model_name, output, **train_cfg)
    typer.echo(f"Detector weights: {weights}")


@app.command("train-classifier")
def cmd_train_classifier(
    coco: Path = typer.Option(..., "--coco"),
    images: Path = typer.Option(..., "--images"),
    class_name: str = typer.Option(..., "--class", help="bag or bike"),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    cfg = _load_config(config).get("classifier", {})
    path = train_classifier(coco, images, class_name, output, **cfg)
    typer.echo(f"Classifier saved: {path}")


@app.command("infer")
def cmd_infer(
    model: Path = typer.Option(..., "--model", help="Detector weights path"),
    images: Path = typer.Option(..., "--images", help="Images directory or file"),
    out: Path = typer.Option(..., "--out", help="Output COCO json path"),
    bag_clf: Optional[Path] = typer.Option(None, "--bag-clf"),
    bike_clf: Optional[Path] = typer.Option(None, "--bike-clf"),
    config: Optional[Path] = typer.Option(None, "--config"),
    tile_for_plate: bool = typer.Option(False, "--tile-for-plate"),
    model_name: str = typer.Option("yolo11n.pt", "--model-name"),
) -> None:
    cfg = _load_config(config)
    det_cfg = cfg.get("detector", {})
    ocr_cfg = cfg.get("ocr", {})

    from neural_cvat.models.yolo_detector import YoloDetector

    if model.suffix == ".pt" or str(model).endswith(".pt"):
        detector = YoloDetector.load(model, **det_cfg)
    elif model.is_dir():
        detector = build_detector(model_name, **det_cfg)
        detector = type(detector).load(model, **det_cfg)
    else:
        detector = build_detector(str(model), **det_cfg)

    bag = AttributeClassifier.load(bag_clf) if bag_clf else None
    bike = AttributeClassifier.load(bike_clf) if bike_clf else None
    pipeline = InferencePipeline(
        detector,
        plate_ocr=PlateOCR(ocr_cfg),
        bag_clf=bag,
        bike_clf=bike,
        tile_for_plate=tile_for_plate or det_cfg.get("tile_for_plate", False),
    )

    if images.is_file():
        paths = [images]
    else:
        paths = sorted(
            list(images.glob("*.jpg")) + list(images.glob("*.jpeg")) + list(images.glob("*.png")),
        )

    pipeline.run(paths, out)
    typer.echo(f"Predictions saved: {out}")


@app.command("download-model")
def download_model(
    model_name: str = typer.Argument(..., help="e.g. yolo11n.pt or HuggingFace repo id"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    if model_name.endswith(".pt") or model_name.lower().startswith("yolo"):
        from ultralytics import YOLO

        model = YOLO(model_name)
        path = output or Path(model_name)
        if output:
            import shutil

            src = getattr(model, "ckpt_path", None) or model_name
            shutil.copy2(src, path)
        typer.echo(f"YOLO model ready: {path if output else model_name}")
    else:
        from transformers import AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(model_name)
        save_dir = output or Path("models") / model_name.replace("/", "_")
        model.save_pretrained(save_dir)
        typer.echo(f"HF model saved: {save_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
