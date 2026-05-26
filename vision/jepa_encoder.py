from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNNEncoder(nn.Module):
    """Small CNN encoder. Outputs a spatial feature map [B, D, 8, 8] (~600K params).

    For I-JEPA pre-training the feature map is used directly.
    For supervised fine-tuning global-average-pooled to [B, D].
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 192):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 160, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, self.feature_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.conv(x)
        return fm.mean(dim=[2, 3])


class IJEPAPretrainer(nn.Module):
    """I-JEPA pre-training on spatial feature maps.

    Predicts target encoder projections at *masked* spatial positions
    from context encoder features using 1x1 conv layers.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        *,
        proj_dim: int = 128,
        mask_ratio: float = 0.3,
        block_size: int = 8,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.block_size = int(block_size)
        self.momentum = float(momentum)
        self.feature_dim = int(feature_dim)

        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.target_projector = nn.Sequential(
            nn.Conv2d(feature_dim, proj_dim, 1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(feature_dim, proj_dim, 1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )

    @torch.no_grad()
    def update_target(self) -> None:
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            tgt_p.data = self.momentum * tgt_p.data + (1.0 - self.momentum) * ctx_p.data

    def random_block_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        bs = self.block_size
        h_blocks, w_blocks = H // bs, W // bs
        n_total = h_blocks * w_blocks
        n_mask = max(1, int(n_total * self.mask_ratio))

        mask_blocks = torch.ones(B, 1, h_blocks, w_blocks, device=x.device)
        for b in range(B):
            idx = torch.randperm(n_total, device=x.device)[:n_mask]
            mask_blocks[b, 0].view(-1)[idx] = 0.0
        mask_pixels = F.interpolate(mask_blocks, size=(H, W), mode="nearest")
        return x * mask_pixels, mask_pixels

    @staticmethod
    def _mask_loss_weight(mask_pixels: torch.Tensor, fm_h: int, fm_w: int) -> torch.Tensor:
        down = F.adaptive_avg_pool2d(mask_pixels, (fm_h, fm_w))
        return (down < 0.5).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_masked, mask_pixels = self.random_block_mask(x)

        ctx_fm = self.context_encoder.forward_feature_map(x_masked)
        pred_fm = self.predictor(ctx_fm)

        with torch.no_grad():
            tgt_fm = self.target_encoder.forward_feature_map(x)
            tgt_proj = self.target_projector(tgt_fm)

        weight = self._mask_loss_weight(mask_pixels, tgt_fm.shape[2], tgt_fm.shape[3])
        diff = (pred_fm - tgt_proj).pow(2).mean(dim=1, keepdim=True)
        loss = (diff * weight).sum() / (weight.sum() + 1e-8)

        self.update_target()
        return loss

    def extract_encoder(self) -> nn.Module:
        return self.context_encoder
