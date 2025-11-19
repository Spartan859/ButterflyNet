#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import os

# Add the project root to sys.path to enable imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.butterfly_net import create_model


def try_torchinfo(model, input_size, out_dir: Path, name: str):
    try:
        from torchinfo import summary  # type: ignore
        info = summary(model, input_size=input_size, verbose=0, col_names=("input_size","output_size","num_params"))
        text_path = out_dir / f"{name}_summary.txt"
        text_path.write_text(str(info), encoding="utf-8")
        print(f"[torchinfo] Saved summary: {text_path}")
    except Exception as e:
        print(f"[torchinfo] Skipped ({e})")


def try_torchviz(model, dummy, out_dir: Path, name: str):
    try:
        from torchviz import make_dot  # type: ignore
        g = make_dot(model(dummy), params=dict(model.named_parameters()))
        # Save dot source
        dot_path = out_dir / f"{name}_graph.dot"
        dot_path.write_text(g.source, encoding="utf-8")
        print(f"[torchviz] Saved DOT: {dot_path}")
        # Attempt render to PNG (requires graphviz installed); fallback if fails
        try:
            png_path = out_dir / f"{name}_graph"
            g.render(str(png_path), format="png", cleanup=True)
            print(f"[torchviz] Rendered PNG: {png_path}.png")
        except Exception as re:
            print(f"[torchviz] Render PNG failed ({re}); DOT retained.")
    except Exception as e:
        print(f"[torchviz] Skipped ({e})")


def ascii_architecture_baseline(out_dir: Path):
    arch = """
Baseline ButterflyNet (input 3x224x224)
  [Conv3x3 3->32] + BN + ReLU + MaxPool2  -> 32x112x112
  [Conv3x3 32->64] + BN + ReLU + MaxPool2 -> 64x56x56
  [Conv3x3 64->128] + BN + ReLU + MaxPool2->128x28x28
  AdaptiveAvgPool(1x1) -> 128
  (Dropout p=?) optional
  Linear 128->50 (logits)
""".strip()
    (out_dir / "baseline_ascii.txt").write_text(arch, encoding="utf-8")


def ascii_architecture_deep(out_dir: Path):
    arch = """
Deep ButterflyNet (input 3x224x224)
  [Conv3x3 3->32] + BN + ReLU + MaxPool2   -> 32x112x112
  [Conv3x3 32->64] + BN + ReLU + MaxPool2  -> 64x56x56
  [Conv3x3 64->128] + BN + ReLU + MaxPool2 ->128x28x28
  [Conv3x3 128->256] + BN + ReLU + MaxPool2->256x14x14
  [Conv3x3 256->512] + BN + ReLU + MaxPool2->512x7x7
  AdaptiveAvgPool(1x1) -> 512
  (Dropout p=?) optional
  Linear 512->50 (logits)
""".strip()
    (out_dir / "deep_ascii.txt").write_text(arch, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate ButterflyNet model diagrams (baseline & deep)")
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/figures"))
    parser.add_argument("--dropout_p", type=float, default=0.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Baseline
    baseline = create_model(num_classes=50, dropout_p=args.dropout_p, model_variant="baseline").to(device)
    try_torchinfo(baseline, (1,3,224,224), args.out_dir, "baseline")
    try_torchviz(baseline, dummy, args.out_dir, "baseline")
    ascii_architecture_baseline(args.out_dir)

    # Deep
    deep = create_model(num_classes=50, dropout_p=args.dropout_p, model_variant="deep").to(device)
    try_torchinfo(deep, (1,3,224,224), args.out_dir, "deep")
    try_torchviz(deep, dummy, args.out_dir, "deep")
    ascii_architecture_deep(args.out_dir)

    print("Done generating model diagrams & summaries in", args.out_dir)


if __name__ == "__main__":
    main()
