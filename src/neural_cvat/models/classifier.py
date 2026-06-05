from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neural_cvat.types import BAG_ATTRIBUTES, BIKE_ATTRIBUTES


class CropAttrDataset(Dataset):
    def __init__(self, crops_dir: Path, attr_names: tuple[str, ...], image_size: int = 224) -> None:
        self.crops_dir = Path(crops_dir)
        meta = json.loads((self.crops_dir / "meta.json").read_text(encoding="utf-8"))
        self.items = meta
        self.attr_names = attr_names
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        path = self.crops_dir / item["file"]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels = torch.tensor([float(item[name]) for name in self.attr_names])
        return image, labels


class AttributeClassifier:
    ATTRS = {"bag": BAG_ATTRIBUTES, "bike": BIKE_ATTRIBUTES}

    def __init__(
        self,
        class_name: str,
        model: nn.Module,
        attr_names: tuple[str, ...],
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        if class_name not in self.ATTRS:
            raise ValueError(f"Unknown class_name: {class_name}")
        self.class_name = class_name
        self.model = model
        self.attr_names = attr_names
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def create(
        cls,
        class_name: str,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        image_size: int = 224,
    ) -> AttributeClassifier:
        import timm

        attr_names = cls.ATTRS[class_name]
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=len(attr_names),
        )
        return cls(class_name, model, attr_names, image_size=image_size)

    def fit(
        self,
        crops_dir: Path | str,
        output_path: Path | str,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Path:
        crops_dir = Path(crops_dir)
        output_path = Path(output_path)
        dataset = CropAttrDataset(crops_dir, self.attr_names, self.image_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        epoch_bar = tqdm(range(epochs), desc=f"{self.class_name} train", unit="epoch")
        for _ in epoch_bar:
            batch_bar = tqdm(loader, desc="batches", leave=False, unit="batch")
            epoch_loss = 0.0
            n_batches = 0
            for images, labels in batch_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            if n_batches:
                epoch_bar.set_postfix(avg_loss=f"{epoch_loss / n_batches:.4f}")

        self.save(output_path)
        return output_path

    def predict(self, crop: np.ndarray) -> dict[str, bool]:
        self.model.eval()
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        return {name: bool(probs[i] >= 0.5) for i, name in enumerate(self.attr_names)}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_name": self.class_name,
                "attr_names": self.attr_names,
                "image_size": self.image_size,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, backbone: str = "efficientnet_b0") -> AttributeClassifier:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls.create(
            ckpt["class_name"],
            backbone=backbone,
            pretrained=False,
            image_size=ckpt["image_size"],
        )
        clf.model.load_state_dict(ckpt["state_dict"])
        return clf
