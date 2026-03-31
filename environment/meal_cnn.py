"""Small CNN encoder for meal images and synthetic image generation for tests."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def synthetic_meal_image(
    height: int = 64,
    width: int = 64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a random RGB meal-like image tensor in NCHW form (1, 3, H, W).
    Uses smooth blobs and light noise so encoders have non-trivial structure in demos.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    c, h, w = 3, height, width
    y = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
    x = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
    img = torch.zeros(c, h, w)
    for _ in range(4):
        cx = float(torch.empty(1).uniform_(-0.7, 0.7))
        cy = float(torch.empty(1).uniform_(-0.7, 0.7))
        sig = float(torch.empty(1).uniform_(0.15, 0.45))
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        blob = torch.exp(-r2 / (2 * sig * sig))
        color = torch.rand(3, 1, 1)
        img = img + blob.unsqueeze(0) * color
    noise = torch.randn(c, h, w) * 0.05
    img = torch.clamp(img + noise, 0.0, 1.0)
    return img.unsqueeze(0)


def synthetic_meal_tensor_from_food_id(
    food_id: int, height: int = 64, width: int = 64
) -> torch.Tensor:
    """Synthetic meal tensor for a food id (deterministic seed per id)."""
    return synthetic_meal_image(
        height=height, width=width, seed=int(food_id) * 9973 + 42
    )


class MealEncoderCNN(nn.Module):
    """
    Lightweight encoder mapping RGB images to an embedding vector (default 64-dim).
    """

    def __init__(self, embedding_dim: int = 64, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, embedding_dim)
        self.macro_head = nn.Linear(embedding_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.head(z)

    def macro_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.macro_head(self.forward(x)), dim=-1)

    @torch.inference_mode()
    def encode_food_id(
        self, food_id: int, device: Optional[torch.device] = None
    ) -> np.ndarray:
        dev = device if device is not None else next(self.parameters()).device
        self.eval()
        self.to(dev)
        x = synthetic_meal_tensor_from_food_id(food_id).to(dev)
        p = self.macro_probs(x)
        return p.squeeze(0).detach().cpu().numpy().astype(np.float32)
