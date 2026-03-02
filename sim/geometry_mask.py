from __future__ import annotations

import numpy as np


def _triangle_mask(px: np.ndarray, py: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    h_tri = np.sqrt(3.0) * 0.5 * side
    v1 = (cx, cy + 2.0 * h_tri / 3.0)
    v2 = (cx - side / 2.0, cy - h_tri / 3.0)
    v3 = (cx + side / 2.0, cy - h_tri / 3.0)

    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, v1[0], v1[1], v2[0], v2[1])
    d2 = sign(px, py, v2[0], v2[1], v3[0], v3[1])
    d3 = sign(px, py, v3[0], v3[1], v1[0], v1[1])

    has_neg = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    has_pos = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return ~(has_neg & has_pos)


def obstacle_mask(px: np.ndarray, py: np.ndarray, shape: str, cx: float, cy: float, d: float) -> np.ndarray:
    if shape == "circle":
        r = 0.5 * d
        return (px - cx) ** 2 + (py - cy) ** 2 <= r**2
    if shape == "square":
        return (np.abs(px - cx) <= 0.5 * d) & (np.abs(py - cy) <= 0.5 * d)
    if shape == "triangle":
        return _triangle_mask(px, py, cx=cx, cy=cy, side=d)
    raise ValueError(f"Unsupported shape: {shape}")


def render_case_image(
    *,
    shape: str,
    dy: float,
    eps: float,
    h: float,
    d_ratio: float,
    x0: float,
    y0: float,
    l_in: float,
    l_out: float,
    image_height: int,
    image_width: int,
    eps_max_for_canvas: float,
) -> np.ndarray:
    """
    Render a synthetic geometry image used as reconstruction target.

    Pixel values:
    - 0.0 : fluid region
    - 0.35: outside channel walls/lens deformation envelope
    - 1.0 : obstacle solid
    """
    l_total = l_in + l_out
    d = d_ratio * h

    h_canvas = h * (1.0 + abs(eps_max_for_canvas))
    y_center = 0.5 * h
    y_min = y_center - 0.5 * h_canvas
    y_max = y_center + 0.5 * h_canvas

    x = (np.arange(image_width, dtype=float) + 0.5) / image_width * l_total
    y = (np.arange(image_height, dtype=float) + 0.5) / image_height * (y_max - y_min) + y_min
    px, py = np.meshgrid(x, y)

    x_transition = l_total - h
    frac = np.clip((px - x_transition) / h, 0.0, 1.0)
    h_local = h * (1.0 + eps * frac)
    y_bottom = y_center - 0.5 * h_local
    y_top = y_center + 0.5 * h_local

    outside = (py < y_bottom) | (py > y_top)
    image = np.zeros((image_height, image_width), dtype=float)
    image[outside] = 0.35

    obs = obstacle_mask(
        px=px,
        py=py,
        shape=shape,
        cx=x0,
        cy=y0 + dy,
        d=d,
    )
    image[obs] = 1.0
    return image

