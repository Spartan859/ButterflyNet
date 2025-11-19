#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_history(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def plot(history, out_path: Path, title: str = 'Training Curves'):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    lr = [h['lr'] for h in history]

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    # Loss subplot
    axes[0].plot(epochs, train_loss, label='Train Loss', color='#1f77b4')
    axes[0].plot(epochs, val_loss, label='Val Loss', color='#ff7f0e')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(title)

    # Accuracy subplot
    axes[1].plot(epochs, train_acc, label='Train Acc', color='#2ca02c')
    axes[1].plot(epochs, val_acc, label='Val Acc', color='#d62728')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # LR subplot (log scale if wide range)
    axes[2].plot(epochs, lr, label='Learning Rate', color='#9467bd')
    axes[2].set_ylabel('LR')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    if max(lr) / min(lr) > 50:
        axes[2].set_yscale('log')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print('Saved figure to', out_path)


def main():
    parser = argparse.ArgumentParser(description='Plot training history curves.')
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('analysis/figures/training_curves.png'))
    parser.add_argument('--title', type=str, default='ButterflyNet Baseline Training')
    args = parser.parse_args()

    history = load_history(args.history)
    plot(history, args.out, args.title)


if __name__ == '__main__':
    main()