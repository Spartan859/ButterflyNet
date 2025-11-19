from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import ButterflyDataset, load_class_mapping
from src.data.transforms import get_transforms_from_stats, get_base_transforms
from src.models.butterfly_net import create_model


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset_root: Path,
    csv_file: Path,
    stats_path: Path | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if stats_path and stats_path.exists():
        transform = get_transforms_from_stats(stats_path)
    else:
        transform = get_base_transforms()
    ds = ButterflyDataset(dataset_root, csv_file, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(checkpoint: Path, num_classes: int, device: torch.device, model_variant: str = "auto") -> nn.Module:
    data = torch.load(checkpoint, map_location=device)
    # Determine variant & dropout from checkpoint metadata when available
    if isinstance(data, dict):
        ckpt_variant = data.get("model_variant", None)
        ckpt_dropout = float(data.get("dropout_p", 0.0)) if data.get("dropout_p", None) is not None else 0.0
        state_dict = data.get("model_state", data)
    else:
        ckpt_variant = None
        ckpt_dropout = 0.0
        state_dict = data

    if model_variant == "auto":
        resolved_variant = ckpt_variant or "baseline"
    else:
        resolved_variant = model_variant

    model = create_model(num_classes=num_classes, dropout_p=ckpt_dropout, model_variant=resolved_variant)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device, limit_batches: int | None = None) -> Tuple[list, list]:
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader):
            if limit_batches is not None and b_idx >= limit_batches:
                break
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())
    return all_preds, all_labels


def compute_metrics(preds: list[int], labels: list[int]) -> Dict[str, object]:
    acc = accuracy_score(labels, preds)
    # Weighted (averaged) metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    # Per-class metrics for potential detailed analysis
    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    report = classification_report(labels, preds, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "support_total": len(labels),
        "per_class_precision": precision_c.tolist(),
        "per_class_recall": recall_c.tolist(),
        "per_class_f1": f1_c.tolist(),
        "per_class_support": support_c.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, normalize: bool = True):
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Row-Normalized)" if normalize else ""))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_metrics(metrics: Dict[str, object], out_json: Path, out_report: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
    with out_report.open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ButterflyNet model on validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("ButterflyClassificationDataset"))
    parser.add_argument("--val_csv", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--class_mapping", type=Path, default=Path("splits/class_to_idx.json"))
    parser.add_argument("--stats", type=Path, default=Path("src/data/dataset_stats.json"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--model_variant", type=str, default="auto", choices=["auto", "baseline", "deep"], help="Model variant. 'auto' tries to read from checkpoint metadata.")
    parser.add_argument("--out_json", type=Path, default=Path("analysis/metrics/val_metrics.json"))
    parser.add_argument("--out_report", type=Path, default=Path("analysis/metrics/classification_report.txt"))
    parser.add_argument("--out_cm", type=Path, default=Path("analysis/figures/confusion_matrix.png"))
    parser.add_argument("--no_normalize_cm", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)
    class_mapping = load_class_mapping(args.class_mapping)
    # Sort class names by index for consistent CM labeling
    inv_map = [None] * len(class_mapping)
    for name, idx in class_mapping.items():
        inv_map[idx] = name

    loader = build_loader(
        args.dataset_root,
        args.val_csv,
        args.stats if args.stats.exists() else None,
        args.batch_size,
        args.num_workers,
    )

    model = load_checkpoint(args.checkpoint, num_classes=len(class_mapping), device=device, model_variant=args.model_variant)
    preds, labels = run_inference(model, loader, device, limit_batches=args.limit_val_batches)
    metrics = compute_metrics(preds, labels)

    print("Accuracy:", metrics["accuracy"])
    print("Weighted Precision:", metrics["precision_weighted"])
    print("Weighted Recall:", metrics["recall_weighted"])
    print("Weighted F1:", metrics["f1_weighted"])
    print(metrics["classification_report"])

    save_metrics(metrics, args.out_json, args.out_report)
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        inv_map,
        args.out_cm,
        normalize=not args.no_normalize_cm,
    )
    print("Saved metrics to", args.out_json)
    print("Saved confusion matrix to", args.out_cm)


if __name__ == "__main__":
    main()