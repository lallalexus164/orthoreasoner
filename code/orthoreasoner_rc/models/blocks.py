from typing import cast

import torch
from torch import Tensor, nn


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, image: Tensor) -> Tensor:
        return cast(Tensor, self.net(image).flatten(1))


class ClinicalEncoder(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, clinical: Tensor) -> Tensor:
        return cast(Tensor, self.net(clinical))


class FusionGate(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.mix = nn.Linear(hidden * 2, hidden)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        joint = torch.cat([left, right], dim=1)
        gate = cast(Tensor, self.gate(joint))
        mixed = cast(Tensor, self.mix(joint))
        return gate * mixed + (1.0 - gate) * (left + right) * 0.5


class TemperatureHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, 1)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        temp = torch.exp(self.log_temperature).clamp(0.25, 8.0)
        return cast(Tensor, self.linear(features).squeeze(1) / temp)
