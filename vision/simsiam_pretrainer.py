"""Lightweight self-supervised methods: SimSiam pre-training for wake field encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamPretrainer(nn.Module):
    """SimSiam self-supervised pre-training for CNN encoder.

    Two augmented views → encoder → projector → predictor.
    Stop-gradient on one branch. Negative cosine similarity loss.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        *,
        proj_dim: int = 256,
        pred_dim: int = 128,
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.noise_std = noise_std
        self.flip_prob = flip_prob

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.flip_prob > 0:
            flip_mask = torch.rand(x.shape[0], device=x.device) < self.flip_prob
            for b in range(x.shape[0]):
                if flip_mask[b]:
                    x[b] = torch.flip(x[b], dims=[-2])
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v1 = self._augment(x)
        v2 = self._augment(x)

        z1 = self.projector(self.encoder(v1))
        z2 = self.projector(self.encoder(v2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = 0.5 * self._cosine_loss(p1, z2.detach()) + 0.5 * self._cosine_loss(p2, z1.detach())
        return loss

    @staticmethod
    def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def extract_encoder(self) -> nn.Module:
        return self.encoder
