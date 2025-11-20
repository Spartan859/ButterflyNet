#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.butterfly_net import create_model
from src.data.transforms import (
    get_transforms_from_stats,
    get_base_transforms,
)


def load_class_mapping(mapping_path: Path) -> dict:
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(checkpoint: Path, num_classes: int, device: torch.device, model_variant: str = "auto") -> tuple[nn.Module, float, str]:
    data = torch.load(checkpoint, map_location=device)
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
    return model, ckpt_dropout, resolved_variant


def resolve_last_conv_layer(model: nn.Module) -> nn.Module:
    # Heuristic: pick the last nn.Conv2d found in modules order
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # activations: (B, C, H, W), gradients: (B, C, H, W)
        acts = self.activations[0]  # (C, H, W)
        grads = self.gradients[0]   # (C, H, W)
        weights = torch.mean(grads, dim=(1, 2))  # (C,)
        cam = torch.sum(weights[:, None, None] * acts, dim=0)  # (H, W)
        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min() if cam_np.min() != 0 else 0
        if cam_np.max() > 0:
            cam_np /= cam_np.max()
        return cam_np


def denormalize_to_uint8(img_t: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    # img_t: (3,H,W) tensor (normalized)
    img = img_t.clone().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1]  # to HWC BGR for cv2
    return img


def overlay_cam_on_image(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def pick_samples_from_csv(dataset_root: Path, csv_file: Path, k: int = 6) -> List[Tuple[Path, int]]:
    import csv as _csv
    items: List[Tuple[Path, int]] = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            items.append((dataset_root / row["path"], int(row["label"])) )
    # deterministic: take first k (csv 已随机并固定seed划分)
    return items[:k]


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualizations for ButterflyNet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("ButterflyClassificationDataset"))
    parser.add_argument("--val_csv", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--class_mapping", type=Path, default=Path("splits/class_to_idx.json"))
    parser.add_argument("--stats", type=Path, default=Path("src/data/dataset_stats.json"))
    parser.add_argument("--model_variant", type=str, default="auto", choices=["auto", "baseline", "deep"])
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/figures/gradcam"))
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = load_class_mapping(args.class_mapping)
    num_classes = len(class_map)

    # transforms (val transforms to match inference pipeline)
    if args.stats.exists():
        tfm = get_transforms_from_stats(args.stats)
        with args.stats.open("r", encoding="utf-8") as f:
            st = json.load(f)
            mean = st.get("mean", [0.485, 0.456, 0.406])
            std = st.get("std", [0.229, 0.224, 0.225])
    else:
        tfm = get_base_transforms()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    model, dropout_p, resolved_variant = load_checkpoint(args.checkpoint, num_classes, device, args.model_variant)
    target_layer = resolve_last_conv_layer(model)
    cam = GradCAM(model, target_layer)

    samples = pick_samples_from_csv(args.dataset_root, args.val_csv, k=args.num_samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, label) in enumerate(samples):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x = tfm(im).unsqueeze(0).to(device)
        cam_map = cam(x, class_idx=None)  # use top-1 predicted class
        img_bgr = denormalize_to_uint8(x[0].cpu(), mean, std)
        overlay = overlay_cam_on_image(img_bgr, cam_map, alpha=args.alpha)

        out_name = f"gradcam_{resolved_variant}_{args.checkpoint.stem}_{idx:02d}.png"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), overlay)
        print("Saved:", out_path)

    cam.remove_hooks()
    print("Done.")


if __name__ == "__main__":
    main()
