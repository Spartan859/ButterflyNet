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


def load_checkpoint(checkpoint_path: Path, device: torch.device, model_variant: str, dropout_p: float, num_classes: int):
    """Load model checkpoint and return model with metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = create_model(num_classes=num_classes, dropout_p=dropout_p, model_variant=model_variant)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return model, {
        'model_variant': model_variant,
        'dropout_p': dropout_p,
        'num_classes': num_classes,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_acc': checkpoint.get('val_acc', 'unknown')
    }


def export_to_onnx(model, dummy_input, onnx_path: Path, opset_version=11):
    """Export PyTorch model to ONNX format."""
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ButterflyNet checkpoint to ONNX format")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the PyTorch checkpoint file")
    parser.add_argument("--output", type=Path, default=None, help="Output ONNX file path (default: same as checkpoint with .onnx extension)")
    parser.add_argument("--model_variant", type=str, default="baseline", choices=["baseline", "deep"], help="Model variant (default: baseline)")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Dropout probability (default: 0.0)")
    parser.add_argument("--num_classes", type=int, default=50, help="Number of classes (default: 50)")
    parser.add_argument("--opset_version", type=int, default=11, help="ONNX opset version (default: 11)")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = args.checkpoint.with_suffix('.onnx')
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, metadata = load_checkpoint(args.checkpoint, device, args.model_variant, args.dropout_p, args.num_classes)
    
    print(f"Model variant: {metadata['model_variant']}")
    print(f"Dropout p: {metadata['dropout_p']}")
    print(f"Num classes: {metadata['num_classes']}")
    print(f"Epoch: {metadata['epoch']}")
    print(f"Best accuracy: {metadata['best_acc']}")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {args.output}")
    export_to_onnx(model, dummy_input, args.output, args.opset_version)
    
    print("ONNX export completed successfully!")


if __name__ == "__main__":
    main()