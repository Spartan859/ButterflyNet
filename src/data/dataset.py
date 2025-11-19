from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Any

from PIL import Image
import torch
from torch.utils.data import Dataset


def default_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ButterflyDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        csv_file: str | Path,
        transform: Optional[Callable] = None,
        return_path: bool = False,
        loader: Callable[[Path], Any] = default_loader,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.return_path = return_path
        self.loader = loader

        assert self.csv_file.exists(), f"Split file not found: {self.csv_file}"
        assert self.dataset_root.exists(), f"Dataset root not found: {self.dataset_root}"

        self.samples = []  # list of (absolute_path, label)
        with self.csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row["path"]
                label = int(row["label"])
                abs_path = (self.dataset_root / rel).resolve()
                self.samples.append((abs_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, label, str(path)
        return img, label


def load_class_mapping(mapping_path: str | Path) -> dict:
    with Path(mapping_path).open("r", encoding="utf-8") as f:
        return json.load(f)
