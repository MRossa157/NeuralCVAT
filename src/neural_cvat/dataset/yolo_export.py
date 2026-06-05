from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from neural_cvat.dataset.coco_io import load_coco
from neural_cvat.types import ID_TO_CATEGORY


def coco_to_yolo(
    coco_path: Path | str,
    images_dir: Path | str,
    out_dir: Path | str,
    splits: dict[str, Path | str] | None = None,
) -> Path:
    """Convert COCO to YOLO format. If splits given, expects {train: path, val: path}."""
    out_dir = Path(out_dir)
    images_dir = Path(images_dir)

    if splits:
        for split_name, split_json in splits.items():
            _export_split(Path(split_json), images_dir, out_dir / split_name)
        data_yaml = _write_data_yaml(out_dir, list(splits.keys()))
    else:
        coco_path = Path(coco_path)
        _export_split(coco_path, images_dir, out_dir / "train")
        data_yaml = _write_data_yaml(out_dir, ["train"])

    return data_yaml


def _export_split(coco_path: Path, images_dir: Path, split_dir: Path) -> None:
    data = load_coco(coco_path)
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in data["images"]:
        fname = img["file_name"]
        src = images_dir / fname
        if not src.exists():
            for alt in images_dir.rglob(fname):
                src = alt
                break
        if src.exists():
            shutil.copy2(src, img_dir / fname)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            cat_name = ID_TO_CATEGORY.get(ann["category_id"])
            if cat_name is None:
                continue
            class_idx = ann["category_id"] - 1
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = lbl_dir / (Path(fname).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_data_yaml(out_dir: Path, split_names: list[str]) -> Path:
    names = ["plate", "car", "bag", "bike"]
    cfg = {
        "path": str(out_dir.resolve()),
        "nc": len(names),
        "names": names,
    }
    for name in split_names:
        cfg[name] = str((out_dir / name / "images").resolve())
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return yaml_path
