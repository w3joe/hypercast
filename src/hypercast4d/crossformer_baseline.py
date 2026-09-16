"""Isolated Crossformer baseline search preceding the HyperDense site sweep."""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from .upstream_models import UpstreamSequence


PROTOCOL = "crossformer-baseline-search-v1"
MODES = ("relative_residual", "levels_direct")


def seed_stream(seed: int, role: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL}/{seed}/{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class CrossformerBaseline(nn.Module):
    def __init__(self, *, window: int, horizon: int = 5, features: int = 7,
                 mode: str = "relative_residual", d_model: int = 32,
                 num_layers: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if window not in (32, 96, 192) or horizon != 5 or features != 7:
            raise ValueError("Unsupported task shape")
        if mode not in MODES or d_model not in (32, 64) or num_layers != 1:
            raise ValueError("Unsupported Crossformer configuration")
        if d_model % n_heads or dropout not in (0.0, 0.1):
            raise ValueError("Invalid model settings")
        self.window, self.horizon, self.features, self.mode = window, horizon, features, mode
        self.sequence = UpstreamSequence("crossformer", window, features, d_model=d_model,
                                         num_layers=num_layers, n_heads=n_heads, dropout=dropout)
        self.output = nn.Linear(window * features, horizon)
        if mode == "relative_residual":
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.window, self.features):
            raise ValueError(f"Expected [batch,{self.window},{self.features}]")
        anchor = x[:, -1, 0:1]
        z = x - x[:, -1:, :] if self.mode == "relative_residual" else x
        z = self.sequence(z).flatten(1)
        prediction = self.output(z)
        return prediction + anchor if self.mode == "relative_residual" else prediction


def build(job: dict) -> CrossformerBaseline:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_stream(job["seed"], "model"))
        return CrossformerBaseline(window=job["window"], mode=job["mode"],
                                   d_model=job["d_model"], dropout=job["dropout"])

