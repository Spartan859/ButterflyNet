from __future__ import annotations

from pathlib import Path
import json
from typing import Tuple, Dict

from PIL import Image
import torch
from torchvision import transforms

# Default target image size
DEFAULT_SIZE: int = 224
# Fallback mean/std (ImageNet) if project stats not yet computed
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_base_transforms(
    size: int = DEFAULT_SIZE,
    mean: Tuple[float, float, float] | None = None,
    std: Tuple[float, float, float] | None = None,
):
    """
    Return the baseline transforms (Resize -> ToTensor -> Normalize).
    If mean/std not provided will fallback to ImageNet stats.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_aug_transforms(
    size: int = DEFAULT_SIZE,
    mean: Tuple[float, float, float] | None = None,
    std: Tuple[float, float, float] | None = None,
):
    """
    Return augmentation transforms for training:
    RandomResizedCrop -> RandomHorizontalFlip -> RandomRotation -> ColorJitter -> ToTensor -> Normalize
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(3/4, 4/3), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_simple_aug_transforms(
    size: int = DEFAULT_SIZE,
    mean: Tuple[float, float, float] | None = None,
    std: Tuple[float, float, float] | None = None,
):
    """Single-op light augmentation: Resize -> RandomHorizontalFlip -> ToTensor -> Normalize."""
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_dataset_stats(stats_path: str | Path) -> Dict[str, list]:
    """Load dataset channel mean/std from a JSON file."""
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"Stats file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_transforms_from_stats(stats_path: str | Path, size: int = DEFAULT_SIZE):
    stats = load_dataset_stats(stats_path)
    mean = stats.get("mean", IMAGENET_MEAN)
    std = stats.get("std", IMAGENET_STD)
    return get_base_transforms(size=size, mean=mean, std=std)


def get_aug_transforms_from_stats(stats_path: str | Path, size: int = DEFAULT_SIZE):
    """Augmentation transforms using dataset statistics if available."""
    stats = load_dataset_stats(stats_path)
    mean = stats.get("mean", IMAGENET_MEAN)
    std = stats.get("std", IMAGENET_STD)
    return get_aug_transforms(size=size, mean=mean, std=std)


def get_simple_aug_transforms_from_stats(stats_path: str | Path, size: int = DEFAULT_SIZE):
    """Single-op light augmentation version using dataset stats."""
    stats = load_dataset_stats(stats_path)
    mean = stats.get("mean", IMAGENET_MEAN)
    std = stats.get("std", IMAGENET_STD)
    return get_simple_aug_transforms(size=size, mean=mean, std=std)
