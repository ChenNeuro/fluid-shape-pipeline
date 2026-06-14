from __future__ import annotations

from typing import cast

import torch
from torch import nn


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.backbone(x))


class MultiScaleWakeNet(nn.Module):
    def __init__(
        self,
        *,
        n_scales: int,
        in_channels: int,
        n_shapes: int,
        n_re_classes: int,
        fusion_hidden: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_scales = int(n_scales)
        self.in_channels = int(in_channels)
        self.n_shapes = int(n_shapes)
        self.n_re_classes = int(n_re_classes)
        self.fusion_hidden = int(fusion_hidden)
        self.dropout = float(dropout)
        self.encoder = ResNet18Encoder(in_channels=in_channels)

        feature_dim = 512 * self.n_scales
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shape_head = nn.Linear(fusion_hidden, n_shapes)
        self.params_head = nn.Linear(fusion_hidden, 2)
        self.re_head = nn.Linear(fusion_hidden, n_re_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, S, C, H, W]
        features = [self.encoder(x[:, scale_idx]) for scale_idx in range(self.n_scales)]
        fused = self.fusion(torch.cat(features, dim=1))
        return {
            "shape_logits": self.shape_head(fused),
            "params_pred": self.params_head(fused),
            "re_logits": self.re_head(fused),
        }


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    *,
    shape_target: torch.Tensor,
    param_target: torch.Tensor,
    re_target: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    w = loss_weights or {}
    w_shape = float(w.get("shape", 1.0))
    w_params = float(w.get("params", 0.5))
    w_re = float(w.get("re", 0.2))
    loss_shape = nn.functional.cross_entropy(outputs["shape_logits"], shape_target)
    loss_params = nn.functional.mse_loss(outputs["params_pred"], param_target)
    loss_re = nn.functional.cross_entropy(outputs["re_logits"], re_target)
    total = w_shape * loss_shape + w_params * loss_params + w_re * loss_re
    return total, {
        "loss_shape": float(loss_shape.detach().cpu()),
        "loss_params": float(loss_params.detach().cpu()),
        "loss_re": float(loss_re.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


class MultiScaleJEPAModel(nn.Module):
    """Multi-scale wake-field classifier using a lightweight JEPA-pretrained CNN encoder."""

    def __init__(
        self,
        *,
        n_scales: int,
        in_channels: int,
        n_shapes: int,
        n_re_classes: int,
        feature_dim: int = 192,
        fusion_hidden: int = 192,
        dropout: float = 0.15,
        pretrained_encoder: nn.Module | None = None,
        encoder_norm: str = "batch",
    ):
        super().__init__()
        self.n_scales = int(n_scales)
        self.in_channels = int(in_channels)
        self.n_shapes = int(n_shapes)
        self.n_re_classes = int(n_re_classes)
        self.feature_dim = int(feature_dim)
        self.fusion_hidden = int(fusion_hidden)
        self.dropout = float(dropout)
        self.encoder_norm = str(encoder_norm)
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            from vision.jepa_encoder import LightweightCNNEncoder

            self.encoder = LightweightCNNEncoder(
                in_channels=in_channels,
                feature_dim=feature_dim,
                norm=encoder_norm,
            )

        fused_dim = feature_dim * self.n_scales
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shape_head = nn.Linear(fusion_hidden, n_shapes)
        self.params_head = nn.Linear(fusion_hidden, 2)
        self.re_head = nn.Linear(fusion_hidden, n_re_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = [self.encoder(x[:, scale_idx]) for scale_idx in range(self.n_scales)]
        fused = self.fusion(torch.cat(features, dim=1))
        return {
            "shape_logits": self.shape_head(fused),
            "params_pred": self.params_head(fused),
            "re_logits": self.re_head(fused),
        }
