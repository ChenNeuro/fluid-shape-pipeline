from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sim.geometry_mask import render_case_image
from vision.wake_dataset import WakeBundle


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]

VARIANTS = {
    # Distance-parameterized crops: dist{D}h_full
    # dist1.0_full = 1.0h downstream from obstacle, full canvas height
    # dist0.5_full = 0.5h downstream, full canvas height
    # dist2.0_full = 2.0h downstream, full canvas height
    "dist_single_4ch": {
        "scales": ["dist1.0_full"],
        "channels": ["ux", "uy", "speed", "vorticity"],
        "description": "Single distance (1.0h), full-height crop",
    },
    "dist_multi_4ch": {
        "scales": ["dist0.5_full", "dist1.0_full", "dist2.0_full"],
        "channels": ["ux", "uy", "speed", "vorticity"],
        "description": "Multi-distance (0.5h/1.0h/2.0h), all full-height",
    },
}


@dataclass(frozen=True)
class LabelMaps:
    shape_to_idx: dict[str, int]
    idx_to_shape: dict[int, str]
    re_to_idx: dict[int, int]
    idx_to_re: dict[int, int]


def build_label_maps(bundle: WakeBundle) -> LabelMaps:
    shapes = sorted(np.unique(bundle.shapes))
    re_values = sorted(int(value) for value in np.unique(bundle.re_values))
    shape_to_idx = {shape: idx for idx, shape in enumerate(shapes)}
    re_to_idx = {value: idx for idx, value in enumerate(re_values)}
    return LabelMaps(
        shape_to_idx=shape_to_idx,
        idx_to_shape={idx: shape for shape, idx in shape_to_idx.items()},
        re_to_idx=re_to_idx,
        idx_to_re={idx: value for value, idx in re_to_idx.items()},
    )


def stratification_labels(bundle: WakeBundle) -> np.ndarray:
    return np.asarray([f"{shape}_Re{int(re_value)}" for shape, re_value in zip(bundle.shapes, bundle.re_values)], dtype=object)


def compute_stratified_test_n(n_total: int, n_strata: int, requested_ratio: float) -> int:
    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    return min(max(requested_test_n, min_test_n), max_test_n)


def repeated_holdout_split(strata: np.ndarray, *, test_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(strata.shape[0], dtype=int)
    idx_train, idx_test = train_test_split(idx, test_size=test_n, random_state=seed, stratify=strata)
    return np.sort(idx_train), np.sort(idx_test)


def accuracy_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def render_targets(
    *,
    shapes: np.ndarray,
    dy_values: np.ndarray,
    eps_values: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    sim_cfg = cfg["simulation"]
    rec_cfg = cfg["reconstruction"]
    images = []
    for shape, dy, eps in zip(shapes, dy_values, eps_values):
        images.append(
            render_case_image(
                shape=str(shape),
                dy=float(dy),
                eps=float(eps),
                h=float(sim_cfg["H"]),
                d_ratio=float(sim_cfg["d_ratio"]),
                x0=float(sim_cfg["x0"]),
                y0=float(sim_cfg["y0"]),
                l_in=float(sim_cfg["L_in"]),
                l_out=float(sim_cfg["L_out"]),
                image_height=int(rec_cfg["image_height"]),
                image_width=int(rec_cfg["image_width"]),
                eps_max_for_canvas=float(sim_cfg["perturb"]["eps_max"]),
            )
        )
    return np.stack(images, axis=0).astype(np.float32)


def clip_params(dy_pred: np.ndarray, eps_pred: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pert_cfg = cfg["simulation"]["perturb"]
    dy_min = float(pert_cfg.get("dy_min", -0.02))
    dy_max = float(pert_cfg.get("dy_max", 0.02))
    eps_max = float(pert_cfg.get("eps_max", 0.02))
    dy_clip = np.clip(dy_pred.astype(float), dy_min, dy_max)
    eps_clip = np.clip(eps_pred.astype(float), -eps_max, eps_max)
    return dy_clip.astype(np.float32), eps_clip.astype(np.float32)


def obstacle_iou_and_dice(targets: np.ndarray, predictions: np.ndarray, threshold: float) -> dict[str, np.ndarray | float]:
    target_mask = targets >= threshold
    pred_mask = predictions >= threshold

    iou_values = []
    dice_values = []
    for idx in range(target_mask.shape[0]):
        inter = np.logical_and(target_mask[idx], pred_mask[idx]).sum()
        union = np.logical_or(target_mask[idx], pred_mask[idx]).sum()
        denom = target_mask[idx].sum() + pred_mask[idx].sum()
        iou_values.append(float(inter / (union + 1e-9)))
        dice_values.append(float((2.0 * inter) / (denom + 1e-9)))

    iou_arr = np.asarray(iou_values, dtype=float)
    dice_arr = np.asarray(dice_values, dtype=float)
    return {
        "iou_mean": float(np.mean(iou_arr)),
        "dice_mean": float(np.mean(dice_arr)),
        "iou_values": iou_arr,
        "dice_values": dice_arr,
    }
