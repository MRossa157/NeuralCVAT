# NeuralCVAT

Пайплайн для дообучения детектора на разметке COCO 1.0 (plate, car, bag, bike), OCR номеров (RapidOCR) и классификаторов атрибутов bag/bike. Результат инференса — COCO 1.0 для импорта в CVAT.

## Установка

```bash
cd NeuralCVAT
pip install -e .
# опционально: SAHI-тайлинг для мелких plate
pip install -e ".[sahi]"
```

## Структура данных

Ожидается формат CVAT COCO 1.0 backup:

```
backup_coco_1.0_216/
├── annotations/instances_default.json
└── images/default/*.jpg
```

Классы: `plate` (1), `car` (2), `bag` (3), `bike` (4).

Атрибуты:
- **plate**: `plate_number`
- **bag**: `is_clean`, `is_deptrans_format`, `is_defected`
- **bike**: `is_good_appearance`

## 1. Подготовка датасета (3 разметчика)

```bash
neural-cvat prepare-dataset \
  images/backup_user1 images/backup_user2 images/backup_user3 \
  --out data/merged \
  --val-ratio 0.15
```

Создаёт `data/merged/annotations/instances_default.json`, копирует изображения и сплиты `data/merged/splits/train.json`, `val.json`.

## 2. Скачать базовую модель

```bash
neural-cvat download-model yolo11n.pt
# или HuggingFace:
neural-cvat download-model PekingU/rtdetr_r50vd --output models/rtdetr
```

## 3. Обучение детектора

YOLO (рекомендуется, `imgsz=1280` для мелких plate):

```bash
neural-cvat train-detector \
  --coco-train data/merged/splits/train.json \
  --coco-val data/merged/splits/val.json \
  --images data/merged/images/default \
  --model-name yolo11n.pt \
  --output runs/detector \
  --epochs 50 --imgsz 1280
```

Конфиг по умолчанию: [`configs/default.yaml`](configs/default.yaml).

## 4. Обучение классификаторов атрибутов

```bash
neural-cvat train-classifier \
  --coco data/merged/splits/train.json \
  --images data/merged/images/default \
  --class bag \
  --output runs/classifiers/bag.pth

neural-cvat train-classifier \
  --coco data/merged/splits/train.json \
  --images data/merged/images/default \
  --class bike \
  --output runs/classifiers/bike.pth
```

## 5. Инференс → COCO 1.0

```bash
neural-cvat infer \
  --model runs/detector/detector/weights/best.pt \
  --images path/to/new_images \
  --bag-clf runs/classifiers/bag.pth \
  --bike-clf runs/classifiers/bike.pth \
  --out predictions/instances_default.json \
  --tile-for-plate
```

Импорт в CVAT: **Project → Import dataset → COCO 1.0**.

## OCR

Модуль [`src/neural_cvat/models/plate_ocr.py`](src/neural_cvat/models/plate_ocr.py) — копия `AutoCVAT/nodes/PlateOCR.py`:
RapidOCR + preprocess (~20 вариантов кропа) + regex + voting.

OCR запускается **только при наличии bbox класса plate** — поэтому критичен recall детектора (низкий `plate_conf`, высокий `imgsz`, опция `--tile-for-plate`).

## Python API

```python
from pathlib import Path
from neural_cvat.inference.pipeline import InferencePipeline
from neural_cvat.models.detector import build_detector
from neural_cvat.models.plate_ocr import PlateOCR
from neural_cvat.models.classifier import AttributeClassifier

detector = build_detector("runs/detector/detector/weights/best.pt")
pipeline = InferencePipeline(
    detector,
    plate_ocr=PlateOCR(),
    bag_clf=AttributeClassifier.load("runs/classifiers/bag.pth"),
    bike_clf=AttributeClassifier.load("runs/classifiers/bike.pth"),
    tile_for_plate=True,
)
coco = pipeline.run(list(Path("images").glob("*.jpg")), "out.json")
```
