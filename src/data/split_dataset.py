import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def find_classes(dataset_root: Path) -> Tuple[List[str], Dict[str, int]]:
    class_names = [p.name for p in dataset_root.iterdir() if p.is_dir()]
    class_names.sort()  # deterministic mapping
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    return class_names, class_to_idx


def collect_images(dataset_root: Path, class_to_idx: Dict[str, int]) -> Dict[int, List[str]]:
    per_class: Dict[int, List[str]] = {idx: [] for idx in class_to_idx.values()}
    for cls_name, idx in class_to_idx.items():
        class_dir = dataset_root / cls_name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix in ALLOWED_EXTS:
                # store relative path (posix) for portability
                rel = p.relative_to(dataset_root).as_posix()
                per_class[idx].append(rel)
    return per_class


def stratified_split(per_class: Dict[int, List[str]], val_ratio: float) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []
    for cls_idx, rel_list in per_class.items():
        rel_list = rel_list.copy()
        random.shuffle(rel_list)
        n = len(rel_list)
        n_val = max(1, int(round(n * val_ratio))) if n > 0 else 0
        val = rel_list[:n_val]
        train = rel_list[n_val:]
        val_items.extend([(p, cls_idx) for p in val])
        train_items.extend([(p, cls_idx) for p in train])
    random.shuffle(train_items)
    random.shuffle(val_items)
    return train_items, val_items


def write_csv(pairs: List[Tuple[str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("path,label\n")
        for rel, lbl in pairs:
            f.write(f"{rel},{lbl}\n")


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val splits for ButterflyClassificationDataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (folder with class subdirs)")
    parser.add_argument("--output-dir", type=str, default="splits", help="Directory to save splits and mapping")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio, e.g., 0.2 for 80/20 split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seeds(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_to_idx = find_classes(dataset_root)
    per_class = collect_images(dataset_root, class_to_idx)
    train_items, val_items = stratified_split(per_class, args.val_ratio)

    write_csv(train_items, output_dir / "train.csv")
    write_csv(val_items, output_dir / "val.csv")

    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    meta = {
        "dataset_root": str(dataset_root),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_classes": len(class_names),
        "num_train": len(train_items),
        "num_val": len(val_items),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Classes: {len(class_names)} | Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
