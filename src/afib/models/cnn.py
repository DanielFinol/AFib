"""Small CNN over the vest leads.

Input (batch, n_leads, time). Leads are treated as channels and convolved over time, then
globally pooled, so the network learns spatial combinations of leads plus temporal/spectral
features, and works for any window length.
"""

from __future__ import annotations

import torch
from torch import nn


class BSPMCNN1D(nn.Module):
    def __init__(
        self,
        n_leads: int,
        n_classes: int,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = n_leads
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def build_model(cfg: dict, n_leads: int, n_classes: int) -> nn.Module:
    name = cfg.get("name", "bspm_cnn1d")
    if name == "bspm_cnn1d":
        return BSPMCNN1D(
            n_leads=n_leads,
            n_classes=n_classes,
            channels=tuple(cfg.get("channels", (32, 64, 128))),
            kernel_size=cfg.get("kernel_size", 7),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(f"unknown model: {name}")
