from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from tqdm import tqdm


def compute_mean_std(dataset_root: Path, csv_file: Path, resize: int = 224) -> dict:
    """Compute per-channel mean and std for images listed in csv_file.

    CSV expected columns: path,label
    Images are loaded converted to RGB and resized to (resize, resize).
    """
    assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"
    assert csv_file.exists(), f"CSV split not found: {csv_file}"

    paths: list[Path] = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["path"]
            p = (dataset_root / rel).resolve()
            if p.exists():
                paths.append(p)

    if len(paths) == 0:
        raise RuntimeError("No images found to compute stats.")

    # Accumulators
    channel_sum = torch.zeros(3)
    channel_sumsq = torch.zeros(3)
    num_pixels_total = 0

    for img_path in tqdm(paths, desc="Computing mean/std"):
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((resize, resize), Image.BICUBIC)
            tensor = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes())).numpy())  # Not efficient; use torchvision ToTensor instead
        # Simpler: use torchvision functional for reliability
        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(resize * resize, 3) / 255.0
        channel_sum += tensor.sum(dim=0)
        channel_sumsq += (tensor ** 2).sum(dim=0)
        num_pixels_total += tensor.shape[0]

    mean = (channel_sum / num_pixels_total).tolist()
    var = (channel_sumsq / num_pixels_total - torch.tensor(mean) ** 2).tolist()
    std = torch.sqrt(torch.tensor(var)).tolist()

    return {"mean": mean, "std": std}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute dataset RGB mean/std from train split.")
    parser.add_argument("--dataset_root", type=Path, default=Path("ButterflyClassificationDataset"))
    parser.add_argument("--train_csv", type=Path, default=Path("splits/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/data/dataset_stats.json"))
    parser.add_argument("--resize", type=int, default=224)
    args = parser.parse_args()

    stats = compute_mean_std(args.dataset_root, args.train_csv, resize=args.resize)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved stats to", args.out)
    print(stats)


if __name__ == "__main__":
    main()
