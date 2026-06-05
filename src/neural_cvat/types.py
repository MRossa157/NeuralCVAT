from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORY_PLATE = 1
CATEGORY_CAR = 2
CATEGORY_BAG = 3
CATEGORY_BIKE = 4

CATEGORY_IDS: dict[str, int] = {
    "plate": CATEGORY_PLATE,
    "car": CATEGORY_CAR,
    "bag": CATEGORY_BAG,
    "bike": CATEGORY_BIKE,
}

ID_TO_CATEGORY: dict[int, str] = {v: k for k, v in CATEGORY_IDS.items()}

COCO_CATEGORIES = [
    {"id": 1, "name": "plate", "supercategory": ""},
    {"id": 2, "name": "car", "supercategory": ""},
    {"id": 3, "name": "bag", "supercategory": ""},
    {"id": 4, "name": "bike", "supercategory": ""},
]

BAG_ATTRIBUTES = ("is_clean", "is_deptrans_format", "is_defected")
BIKE_ATTRIBUTES = ("is_good_appearance",)


@dataclass
class Detection:
    bbox_xywh: tuple[float, float, float, float]
    category_id: int
    score: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class COCOImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    attributes: dict[str, Any] = field(default_factory=dict)
