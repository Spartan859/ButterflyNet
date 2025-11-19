from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # spatial downsample by 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ButterflyNet(nn.Module):
    """Baseline CNN for 50-class butterfly classification.

    Architecture rationale:
    - 3 conv blocks (channels: 3->32->64->128) balance capacity and overfitting risk on medium dataset.
    - Each block: Conv(3x3) + BN + ReLU + MaxPool(2) for stable training & controlled receptive field growth.
    - Final classifier: global average pooling (reduces params vs large fully connected) followed by Linear.
    - Design keeps parameter count modest, facilitating later comparison with deeper/backbone variants.
    """

    def __init__(self, num_classes: int = 50, dropout_p: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),   # -> 32 x 112 x 112 (assuming input 224)
            ConvBlock(32, 64),  # -> 64 x 56 x 56
            ConvBlock(64, 128), # -> 128 x 28 x 28
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


def create_model(num_classes: int = 50, dropout_p: float = 0.0) -> ButterflyNet:
    return ButterflyNet(num_classes=num_classes, dropout_p=dropout_p)


if __name__ == "__main__":
    # Quick shape sanity check
    model = create_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # Expect (2, 50)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))