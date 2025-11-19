from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import ButterflyDataset, load_class_mapping
from src.data.transforms import (
    get_transforms_from_stats,
    get_base_transforms,
    get_aug_transforms,
    get_aug_transforms_from_stats,
    get_simple_aug_transforms,
    get_simple_aug_transforms_from_stats,
)
from src.models.butterfly_net import create_model


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (pred.argmax(dim=1) == target).float().mean().item()


def build_dataloaders(
    dataset_root: Path,
    train_csv: Path,
    val_csv: Path,
    stats_path: Path | None,
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
    use_aug: bool = False,
    aug_mode: str = "full",
) -> Tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    if stats_path and stats_path.exists():
        if use_aug:
            if aug_mode == "simple":
                train_transform = get_simple_aug_transforms_from_stats(stats_path)
            else:
                train_transform = get_aug_transforms_from_stats(stats_path)
        else:
            train_transform = get_transforms_from_stats(stats_path)
        val_transform = get_transforms_from_stats(stats_path)
    else:
        if use_aug:
            if aug_mode == "simple":
                train_transform = get_simple_aug_transforms()
            else:
                train_transform = get_aug_transforms()
        else:
            train_transform = get_base_transforms()
        val_transform = get_base_transforms()

    train_ds = ButterflyDataset(dataset_root, train_csv, transform=train_transform)
    val_ds = ButterflyDataset(dataset_root, val_csv, transform=val_transform)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(not distributed),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler, val_sampler


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, limit_batches: int | None = None):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    count = 0
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader):
            if limit_batches is not None and b_idx >= limit_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_acc += accuracy(out, y) * x.size(0)
            count += x.size(0)
    return {
        "loss": total_loss / max(count, 1),
        "acc": total_acc / max(count, 1),
    }


def train_epoch(model, loader, optimizer, device, limit_batches: int | None = None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for b_idx, (x, y) in enumerate(loader):
        if limit_batches is not None and b_idx >= limit_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        batch_acc = accuracy(out, y)
        total_loss += loss.item() * x.size(0)
        total_acc += batch_acc * x.size(0)
        count += x.size(0)
    return {
        "loss": total_loss / max(count, 1),
        "acc": total_acc / max(count, 1),
    }


def save_checkpoint(state: dict, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save(state, path)
    return path


def init_distributed(device_str: str | None = None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend=backend)
    return is_distributed, rank, local_rank, world_size, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main():
    parser = argparse.ArgumentParser(description="Train ButterflyNet baseline model.")
    parser.add_argument("--dataset_root", type=Path, default=Path("ButterflyClassificationDataset"))
    parser.add_argument("--train_csv", type=Path, default=Path("splits/train.csv"))
    parser.add_argument("--val_csv", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--class_mapping", type=Path, default=Path("splits/class_to_idx.json"))
    parser.add_argument("--stats", type=Path, default=Path("src/data/dataset_stats.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_train_batches", type=int, default=None, help="For dry-run: limit number of train batches per epoch")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="For dry-run: limit number of val batches per eval")
    parser.add_argument("--dry_run", action="store_true", help="Run only one epoch with limited batches to verify pipeline")
    parser.add_argument("--distributed", action="store_true", help="Force enable distributed training (torchrun)")
    parser.add_argument("--aug", action="store_true", help="Use data augmentations for training")
    parser.add_argument("--aug_mode", type=str, default="full", choices=["full", "simple"], help="Augmentation mode: full (multi-op) or simple (single flip)")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Dropout probability applied before classifier (0 to disable)")
    parser.add_argument("--model_variant", type=str, default="baseline", choices=["baseline", "deep"], help="Model architecture variant")
    args = parser.parse_args()

    set_seeds(args.seed)
    # auto detect distributed via WORLD_SIZE>1 unless explicitly disabled
    autodetect_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    use_distributed = args.distributed or autodetect_distributed
    is_distributed, rank, local_rank, world_size, device = init_distributed(args.device if use_distributed else args.device)

    class_mapping = load_class_mapping(args.class_mapping)
    num_classes = len(class_mapping)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        args.dataset_root,
        args.train_csv,
        args.val_csv,
        args.stats if args.stats.exists() else None,
        args.batch_size,
        num_workers=args.num_workers,
        distributed=is_distributed,
        use_aug=args.aug,
        aug_mode=args.aug_mode,
    )

    model = create_model(num_classes=num_classes, dropout_p=args.dropout_p, model_variant=args.model_variant).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    history = []
    best_acc = -1.0
    start = datetime.now().strftime("%Y%m%d_%H%M%S")

    epochs = 1 if args.dry_run else args.epochs
    for epoch in range(1, epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            limit_batches=args.limit_train_batches if args.dry_run else args.limit_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            limit_batches=args.limit_val_batches if args.dry_run else args.limit_val_batches,
        )
        scheduler.step(val_metrics["acc"])

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if is_main_process(rank):
            print(f"Epoch {epoch}/{epochs} | "
                  f"train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} "
                  f"val_loss={record['val_loss']:.4f} val_acc={record['val_acc']:.4f} lr={record['lr']:.2e}")

        if record["val_acc"] > best_acc and is_main_process(rank):
            best_acc = record["val_acc"]
            # unwrap model if DDP
            model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            tag_parts = []
            if args.model_variant and args.model_variant != "baseline":
                tag_parts.append(args.model_variant)
            tag_parts.append(f"aug-{args.aug_mode}" if args.aug else "baseline")
            if args.dropout_p and args.dropout_p > 0.0:
                tag_parts.append(f"drop{args.dropout_p:.2f}")
            tag = "_".join(tag_parts)
            ckpt_path = save_checkpoint({
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_acc,
                "class_mapping": class_mapping,
                "world_size": world_size,
                "aug": args.aug,
                "aug_mode": args.aug_mode,
                "dropout_p": args.dropout_p,
                "model_variant": args.model_variant,
            }, args.out_dir, f"{tag}_best_{start}.pt")
            print(f"Saved new best checkpoint: {ckpt_path}")

        if args.dry_run:
            if is_main_process(rank):
                print("Dry-run mode enabled; stopping after first epoch.")
            break

    # Save training history
    if is_main_process(rank):
        tag_bits = []
        if args.model_variant and args.model_variant != "baseline":
            tag_bits.append(args.model_variant)
        if args.aug:
            tag_bits.append(f"aug-{args.aug_mode}")
        if args.dropout_p and args.dropout_p > 0.0:
            tag_bits.append(f"drop{args.dropout_p:.2f}")
        tag_prefix = ("_".join(tag_bits) + "_") if tag_bits else ""
        hist_path = args.out_dir / f"history_{tag_prefix}{start}.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print("Training history saved to", hist_path)

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()