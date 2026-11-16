"""Frame encoder (ResNet-50) and temporal models over per-second features."""
from __future__ import annotations

import torch
import torch.nn as nn


def frame_encoder(n_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """ImageNet ResNet-50 with a phase-classification head. `forward_features` + pooling give
    the 2048-d embedding used by the temporal model."""
    import timm
    return timm.create_model("resnet50.a1_in1k", pretrained=pretrained, num_classes=n_classes)


def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pooled 2048-d features before the classification head."""
    return model.forward_head(model.forward_features(x), pre_logits=True)


class TemporalLSTM(nn.Module):
    """LSTM over the feature sequence of one video.

    bidirectional=False is causal: the prediction at second t uses only seconds <= t, so it can
    run live in the operating room. bidirectional=True sees the whole video (offline analysis).
    """

    def __init__(self, dim: int = 2048, hidden: int = 512, n_classes: int = 7, bidirectional: bool = False,
                 dropout: float = 0.5):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (B, T, dim)
        h, _ = self.lstm(self.inp(x))
        return self.out(h)                                         # (B, T, n_classes)
