from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class DifferentiableShapeRenderer(nn.Module):
    """
    SDF-based differentiable renderer producing soft obstacle probability maps.
    Uses a clipped-linear transition zone to produce masks that reach [0,1] fully.

    Shape order: 0=circle, 1=triangle, 2=airfoil, 3=diamond, 4=bar
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        transition_px: int = 2,
        l_total: float = 10.0,
        h: float = 1.0,
        d_ratio: float = 0.2,
        x0: float = 3.0,
        y0: float = 0.5,
        eps_max: float = 0.06,
    ):
        super().__init__()
        self.image_size = image_size
        self.transition_px = transition_px
        self.l_total = l_total
        self.h = h
        self.d = d_ratio * h
        self.x0 = x0
        self.y0 = y0
        self.eps_max = eps_max

    def _physics_grid(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h_canvas = self.h * (1.0 + abs(self.eps_max))
        y_center = 0.5 * self.h
        y_min = y_center - 0.5 * h_canvas
        y_max = y_center + 0.5 * h_canvas

        x_phys = (torch.arange(self.image_size, dtype=torch.float32, device=device) + 0.5) / self.image_size * self.l_total
        y_phys = (torch.arange(self.image_size, dtype=torch.float32, device=device) + 0.5) / self.image_size * (y_max - y_min) + y_min

        grid_x, grid_y = torch.meshgrid(x_phys, y_phys, indexing="xy")
        return grid_x, grid_y

    def _edge_sdf(self, px: torch.Tensor, py: torch.Tensor, ax: float, ay: float, bx: float, by: float) -> torch.Tensor:
        import math
        abx = bx - ax
        aby = by - ay
        len_ab = math.sqrt(abx ** 2 + aby ** 2)
        cross_num = (px - ax) * aby - (py - ay) * abx
        return cross_num / (len_ab + 1e-12)

    def _shape_sdf(self, px: torch.Tensor, py: torch.Tensor, shape_idx: int, cy: torch.Tensor) -> torch.Tensor:
        if shape_idx == 0:
            # Circle: Euclidean SDF (negative inside)
            return torch.sqrt((px - self.x0) ** 2 + (py - cy) ** 2 + 1e-12) - 0.5 * self.d

        elif shape_idx == 1:
            # Triangle: equilateral, vertex up, side = d
            h_tri = 0.8660254037844386 * self.d
            v1x = self.x0
            v1y = cy + 2.0 * h_tri / 3.0
            v2x = self.x0 - self.d / 2.0
            v2y = cy - h_tri / 3.0
            v3x = self.x0 + self.d / 2.0
            v3y = cy - h_tri / 3.0

            d1 = self._edge_sdf(px, py, v1x, float(v1y.item()), v2x, float(v2y.item()))
            d2 = self._edge_sdf(px, py, v2x, float(v2y.item()), v3x, float(v3y.item()))
            d3 = self._edge_sdf(px, py, v3x, float(v3y.item()), v1x, float(v1y.item()))
            return torch.max(torch.max(d1, d2), d3)

        elif shape_idx == 2:
            xr = (px - self.x0) / self.d + 0.5
            xr_c = torch.clamp(xr, 0.0, 1.0)
            yt_c = 5.0 * 0.14 * (
                0.2969 * torch.sqrt(xr_c + 1e-12)
                - 0.1260 * xr_c
                - 0.3516 * xr_c ** 2
                + 0.2843 * xr_c ** 3
                - 0.1015 * xr_c ** 4
            )
            half_thickness = self.d * yt_c
            upper_dist = py - (cy + half_thickness)
            lower_dist = (cy - half_thickness) - py
            inside = (py >= (cy - half_thickness)) & (py <= (cy + half_thickness))
            return torch.where(inside, -torch.min(torch.abs(upper_dist), torch.abs(lower_dist)), torch.min(torch.abs(upper_dist), torch.abs(lower_dist)))

        elif shape_idx == 3:
            return torch.abs(px - self.x0) / (0.56 * self.d + 1e-12) + torch.abs(py - cy) / (0.38 * self.d + 1e-12) - 1.0

        elif shape_idx == 4:
            qx = torch.abs(px - self.x0) - 0.75 * self.d
            qy = torch.abs(py - cy) - 0.18 * self.d
            outside = torch.max(qx, qy)
            inside = torch.min(-qx, -qy)
            return torch.where(outside > 0, outside, inside)

        else:
            raise ValueError(f"Unknown shape index: {shape_idx}")

    def _sdf_to_mask(self, sdf: torch.Tensor) -> torch.Tensor:
        # Clipped-linear mask: full inside (sdf<0), full outside (sdf>tw),
        # linear transition in between. Guarantees mask in [0,1].
        tw = self.transition_px * self.l_total / self.image_size
        return torch.clamp(1.0 - sdf / tw, 0.0, 1.0)

    def render(
        self,
        shape_indices: torch.Tensor,
        dy_values: torch.Tensor,
        eps_values: torch.Tensor,
    ) -> torch.Tensor:
        device = shape_indices.device
        px, py = self._physics_grid(device)
        cy_center = self.y0 + dy_values

        B = shape_indices.shape[0]
        masks: list[torch.Tensor] = []
        for b in range(B):
            sdf = self._shape_sdf(px, py, int(shape_indices[b].item()), cy_center[b])
            masks.append(self._sdf_to_mask(sdf))

        return torch.stack(masks, dim=0).unsqueeze(1)

    def soft_iou_loss(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection + 1e-8
        return 1.0 - intersection / union

    def soft_dice_loss(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        intersection = (pred_mask * target_mask).sum()
        return 1.0 - (2.0 * intersection + 1e-8) / (pred_mask.sum() + target_mask.sum() + 1e-8)

    def render_and_loss(
        self,
        shape_pred: torch.Tensor,
        dy_pred: torch.Tensor,
        eps_pred: torch.Tensor,
        shape_target: torch.Tensor,
        dy_target: torch.Tensor,
        eps_target: torch.Tensor,
        metric: Literal["iou", "dice"] = "dice",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_mask = self.render(shape_pred, dy_pred, eps_pred)
        target_mask = self.render(shape_target, dy_target, eps_target)
        loss = self.soft_dice_loss(pred_mask, target_mask) if metric == "dice" else self.soft_iou_loss(pred_mask, target_mask)
        return loss, pred_mask, target_mask

    def eval_soft_metrics(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
        # Binarize with 0.5 threshold so self-check = 1.0 and metrics are interpretable
        pred_bin = (pred_mask > 0.5).float()
        target_bin = (target_mask > 0.5).float()
        intersection = (pred_bin * target_bin).sum()
        pred_sum = pred_bin.sum()
        target_sum = target_bin.sum()
        union = pred_sum + target_sum - intersection + 1e-8
        return {
            "soft_iou": float((intersection / union).detach().cpu()),
            "soft_dice": float(((2.0 * intersection) / (pred_sum + target_sum + 1e-8)).detach().cpu()),
        }