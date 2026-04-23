from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MAEViTEncoder(nn.Module):
    """
    Wraps timm's vit_base_patch16_224.mae for multi-channel [C, H, W] input.
    Forward: [B, C, H, W] → [B, 768] CLS-token feature.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224.mae",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_name = model_name

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
            )
        except Exception:
            # Fallback: manually replace patch embed for multi-channel input
            self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
            old_proj = self.backbone.patch_embed.proj
            self.backbone.patch_embed.proj = nn.Conv2d(
                in_channels, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=False,
            )

        self.feature_dim = getattr(self.backbone, "num_features", 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (e.g. [B, 4, 128, 128])
        B, C, H, W = x.shape
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)

        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x[:, 0]


class MultiScaleViTWakeNet(nn.Module):
    """
    MAE ViT backbone replacing ResNet18 in MultiScaleWakeNet.

    Input:  [B, S, C, 128, 128]  multi-scale crops
    Output: 3 heads (shape_logits, params_pred, re_logits)

    Two-phase training:
      Phase 1 (linear probe): freeze backbone, train projection+fusion+heads
      Phase 2 (fine-tune):    unfreeze with layer-wise LR decay
    """

    def __init__(
        self,
        *,
        n_scales: int,
        in_channels: int = 4,
        n_shapes: int,
        n_re_classes: int,
        fusion_hidden: int = 512,
        proj_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224.mae",
    ):
        super().__init__()
        self.n_scales = n_scales
        self.vit_feature_dim = 768

        self.encoder = MAEViTEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            model_name=model_name,
        )

        self.scale_proj = nn.ModuleList(
            [nn.Linear(self.vit_feature_dim, proj_dim) for _ in range(n_scales)]
        )

        fused_dim = proj_dim * n_scales
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shape_head = nn.Linear(fusion_hidden, n_shapes)
        self.params_head = nn.Linear(fusion_hidden, 2)
        self.re_head = nn.Linear(fusion_hidden, n_re_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, S, C, 128, 128]
        features = []
        for scale_idx in range(self.n_scales):
            crop = x[:, scale_idx]
            cls_token = self.encoder(crop)
            scaled = self.scale_proj[scale_idx](cls_token)
            features.append(scaled)

        fused = self.fusion(torch.cat(features, dim=1))
        return {
            "shape_logits": self.shape_head(fused),
            "params_pred": self.params_head(fused),
            "re_logits": self.re_head(fused),
        }

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_with_llrd(
        self,
        base_lr: float,
        llrd_decay: float = 0.85,
    ) -> list[dict]:
        encoder = self.encoder.backbone

        # Map each parameter to its block depth (0-11 for ViT-B/16)
        block_params: dict[int, list[torch.Tensor]] = {}
        non_block_decay: list[torch.Tensor] = []
        non_block_no_decay: list[torch.Tensor] = []

        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
            if "blocks." in name:
                try:
                    depth = int(name.split("blocks.")[1].split(".")[0])
                    block_params.setdefault(depth, []).append(param)
                except ValueError:
                    pass
            elif "bias" in name or "norm" in name or "bn" in name:
                non_block_no_decay.append(param)
            else:
                non_block_decay.append(param)

        n_blocks = max(max(block_params.keys()) + 1, 12)

        groups: list[dict] = []
        seen_params: set[int] = set()

        def add_group(params: list[torch.Tensor], lr: float, wd: float) -> None:
            for p in params:
                pid = id(p)
                if pid not in seen_params:
                    groups.append({"params": [p], "lr": lr, "weight_decay": wd})
                    seen_params.add(pid)

        # Block layers: deeper blocks (higher depth) get higher LR
        for depth in range(n_blocks):
            if depth in block_params:
                lr = base_lr * (llrd_decay ** (n_blocks - 1 - depth))
                add_group(block_params[depth], lr, 0.01)

        # Patch embed / pos embed / cls token: lowest LR
        lowest_lr = base_lr * (llrd_decay ** n_blocks)
        add_group(non_block_decay, lowest_lr, 0.01)
        add_group(non_block_no_decay, lowest_lr, 0.0)

        # Head parameters: full LR
        head_decay: list[torch.Tensor] = []
        head_no_decay: list[torch.Tensor] = []
        for name, param in self.named_parameters():
            if "encoder.backbone" in name:
                continue
            if param.ndim >= 2 and "bias" not in name:
                head_decay.append(param)
            else:
                head_no_decay.append(param)
        add_group(head_decay, base_lr, 0.01)
        add_group(head_no_decay, base_lr, 0.0)

        return groups