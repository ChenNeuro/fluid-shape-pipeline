"""
Flask web app for real-time wake-field shape classification.
Run:  python web_app/app.py
Open: http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# ── Add project root to Python path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request

# ── Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "configs" / "wake_field_450.yaml"
MODEL_PATH = ROOT / "models" / "wake_field_main.pt"

from ml.wake_common import build_label_maps
from sim.config import load_config
from vision.mae_vit_model import MultiScaleViTWakeNet
from vision.wake_dataset import load_wake_bundle
from vision.wake_model import MultiScaleWakeNet

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
FIELDS_DIR = ROOT / "web_app" / "results"
FIELDS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg: Optional[dict] = None
model: Optional[torch.nn.Module] = None
label_maps = None
MODEL_TYPE = "resnet18"
MODEL_KWARGS = None

# ── Synthetic wake field generator ───────────────────────────────────────

SHAPE_PROFILES = {
    "circle": {
        "decay": 0.54,
        "spread": 0.070,
        "alt_offset": 0.14,
        "deficit": 0.22,
        "lift_bias": 0.00,
    },
    "triangle": {
        "decay": 0.56,
        "spread": 0.074,
        "alt_offset": 0.13,
        "deficit": 0.24,
        "lift_bias": 0.00,
    },
    "airfoil": {
        "decay": 0.50,
        "spread": 0.060,
        "alt_offset": 0.10,
        "deficit": 0.18,
        "lift_bias": 0.05,
    },
    "diamond": {
        "decay": 0.60,
        "spread": 0.078,
        "alt_offset": 0.17,
        "deficit": 0.27,
        "lift_bias": 0.01,
    },
    "bar": {"decay": 0.70, "spread": 0.090, "alt_offset": 0.19, "deficit": 0.33, "lift_bias": 0.00},
}

SHAPE_PARAMS = {
    "circle": {"st": 0.17, "amp": 0.13, "h2": 0.16, "h3": 0.05, "phase_gradient": 1.8},
    "triangle": {"st": 0.30, "amp": 0.20, "h2": 0.34, "h3": 0.18, "phase_gradient": 3.3},
    "airfoil": {"st": 0.11, "amp": 0.09, "h2": 0.08, "h3": 0.03, "phase_gradient": 1.2},
    "diamond": {"st": 0.24, "amp": 0.18, "h2": 0.42, "h3": 0.27, "phase_gradient": 2.8},
    "bar": {"st": 0.07, "amp": 0.24, "h2": 0.58, "h3": 0.42, "phase_gradient": 4.0},
}


def _base_profile(y_norm: np.ndarray) -> np.ndarray:
    return 6.0 * y_norm * (1.0 - y_norm)


def generate_wake_velocity_field(
    shape: str,
    re_val: float,
    dy: float,
    eps: float,
    seed: int = 42,
    canvas_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate [2, H, W] velocity field (ux, uy) for a shape at given params."""
    rng = np.random.default_rng(seed)

    H = 1.0
    d_ratio = 0.2
    d = d_ratio * H
    U_mean = 1.0
    x0 = 3.0
    y0 = 0.5
    L_in = 5.0
    L_out = 5.0
    L_total = L_in + L_out
    params = SHAPE_PARAMS.get(shape, SHAPE_PARAMS["circle"])
    profile = SHAPE_PROFILES.get(shape, SHAPE_PROFILES["circle"])

    # Canvas in physics units
    x_start = x0 + 0.5 * d
    eps_max = 0.06
    y_center = 0.5 * H
    h_canvas = H * (1.0 + eps_max)
    y_min = y_center - 0.5 * h_canvas
    y_max = y_center + 0.5 * h_canvas

    x_phys = (np.arange(canvas_size, dtype=float) + 0.5) / canvas_size * (
        L_total - x_start
    ) + x_start
    y_phys = (np.arange(canvas_size, dtype=float) + 0.5) / canvas_size * (y_max - y_min) + y_min
    x_grid, y_grid = np.meshgrid(x_phys, y_phys)

    # Vortex shedding frequency
    st = float(params["st"])
    f0 = (
        st
        * U_mean
        / d
        * (1.0 + 0.08 * ((re_val - 200.0) / 200.0))
        * (1.0 + 0.25 * eps + 0.10 * (dy / H))
    )
    f0 = max(0.05, f0)

    # Local channel half-height at downstream position
    x_transition = L_total - H
    frac = np.clip((x_grid - x_transition) / H, 0.0, 1.0)
    h_local = H * (1.0 + eps * frac)
    y_bottom = y_center - 0.5 * h_local
    y_top = y_center + 0.5 * h_local
    inside = (y_grid >= y_bottom) & (y_grid <= y_top)

    y_norm = np.clip((y_grid - y_bottom) / (h_local + 1e-9), 1e-4, 1.0 - 1e-4)
    base_u = U_mean * _base_profile(y_norm)

    roi_len = max(1e-6, L_total - x_start)
    xn = np.clip((x_grid - x_start) / roi_len, 0.0, 1.0)

    lift_bias = profile["lift_bias"] * math.sqrt(re_val / 200.0)
    wake_center = y0 + dy + lift_bias * H * np.exp(-1.8 * xn)
    spread = profile["spread"] + 0.05 * xn + 0.02 * abs(eps)
    cross = (y_grid - wake_center) / (H + 1e-9)
    core = np.exp(-(cross**2) / (2.0 * spread**2))
    offset = profile["alt_offset"] + 0.08 * xn
    sigma_pair = spread * 0.8
    upper = np.exp(-((cross - offset) ** 2) / (2.0 * sigma_pair**2))
    lower = np.exp(-((cross + offset) ** 2) / (2.0 * sigma_pair**2))
    pair_sum = upper + lower
    pair_diff = upper - lower

    decay = max(0.25, profile["decay"])
    wake_env = np.exp(-xn / decay)
    phase = 2.0 * np.pi * (0.65 * xn - f0 * 0.5) + float(params["phase_gradient"]) * cross
    harmonic = np.sin(phase)
    harmonic += float(params["h2"]) * np.sin(2.0 * phase + 0.3)
    harmonic += float(params["h3"]) * np.sin(3.0 * phase - 0.2)
    convective = np.cos(phase + 0.25) + 0.35 * np.cos(2.0 * phase - 0.15)

    amp_base = float(params["amp"]) * math.sqrt(re_val / 200.0)
    deficit = profile["deficit"] * wake_env * np.clip(0.7 * core + 0.5 * pair_sum, 0.0, 1.6)

    u_field = base_u * (1.0 - deficit)
    u_field += 0.36 * amp_base * wake_env * (0.35 * core + pair_sum) * convective
    u_field += 0.18 * amp_base * wake_env * cross * np.sin(0.5 * phase + 0.4)
    v_field = 1.10 * amp_base * wake_env * pair_diff * harmonic
    v_field += 0.26 * amp_base * wake_env * core * np.sin(phase + 0.6)
    v_field += 0.08 * amp_base * wake_env * lift_bias

    # Add noise proportional to eps
    noise_std = 0.03 * (1.0 + 2.0 * abs(eps))
    u_field = np.where(inside, u_field, 0.0) + rng.normal(0, noise_std, size=u_field.shape)
    v_field = np.where(inside, v_field, 0.0) + rng.normal(0, noise_std * 0.5, size=v_field.shape)

    roi = {
        "x_min": float(x_start),
        "x_max": float(L_total),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }
    return (np.stack([u_field, v_field], axis=0)).astype(np.float32), roi


def compute_vorticity(vel: np.ndarray) -> np.ndarray:
    """Compute vorticity from [2, H, W] velocity field."""
    ux = vel[0]
    uy = vel[1]
    # Finite difference (boundary-aware)
    omega = np.zeros_like(ux)
    omega[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) * 0.5
    omega[1:-1, :] -= (ux[2:, :] - ux[:-2, :]) * 0.5
    # Forward/backward at edges
    omega[:, 0] = (uy[:, 1] - uy[:, 0]) - (ux[1, 0] - ux[-1 if 0 else 1, 0])
    omega[:, -1] = (uy[:, -1] - uy[:, -2]) - (ux[-1 if -1 else 0, -1] - ux[-2, -1])
    omega[0, :] -= ux[1, :] - ux[0, :]
    omega[-1, :] -= ux[-1, :] - ux[-2, :]
    return omega.astype(np.float32)


def compute_speed(vel: np.ndarray) -> np.ndarray:
    return np.sqrt(vel[0] ** 2 + vel[1] ** 2).astype(np.float32)


def normalize_field(field: np.ndarray) -> np.ndarray:
    channel_mean = field.reshape(field.shape[0], -1).mean(axis=1)
    channel_std = field.reshape(field.shape[0], -1).std(axis=1)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    return ((field - channel_mean[:, None, None]) / channel_std[:, None, None]).astype(np.float32)


def crop_box(
    norm_x0: float,
    norm_y0: float,
    norm_x1: float,
    norm_y1: float,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    def _px(v, n):
        return int(np.clip(int(v * n), 0, n - 1))

    x0 = _px(norm_x0, w)
    x1 = _px(norm_x1, w)
    y0 = _px(norm_y0, h)
    y1 = _px(norm_y1, h)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return x0, y0, x1, y1


def resize_crop(field: np.ndarray, norm_box: list, output_size: int) -> np.ndarray:
    _, h, w = field.shape
    x0, y0, x1, y1 = crop_box(norm_box[0], norm_box[1], norm_box[2], norm_box[3], h, w)
    crop = field[:, y0:y1, x0:x1]
    resized = np.stack(
        [
            cv2.resize(crop[c], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            for c in range(crop.shape[0])
        ],
        axis=0,
    )
    return resized.astype(np.float32)


def build_distance_crop_box(
    downstream_h: float,
    canvas_x_start: float,
    canvas_x_end: float,
    canvas_y_min: float,
    canvas_y_max: float,
) -> list:
    """Build [x0,y0,x1,y1] in normalized canvas coords."""
    h_canvas = canvas_y_max - canvas_y_min
    x_start_phys = canvas_x_start + downstream_h * h_canvas
    x_end_phys = canvas_x_end
    x0_norm = float(
        np.clip((x_start_phys - canvas_x_start) / (canvas_x_end - canvas_x_start), 0.0, 1.0)
    )
    x1_norm = 1.0
    canvas_y_center = (canvas_y_min + canvas_y_max) * 0.5
    y0_norm = 0.0
    y1_norm = 1.0
    return [x0_norm, y0_norm, x1_norm, y1_norm]


def prepare_multiscale_input(
    vel: np.ndarray,
    roi: dict,
    scales: list,
    canvas_size: int = 256,
    output_size: int = 128,
) -> np.ndarray:
    """Build [S, 4, H, W] multi-scale input for the model."""
    vort = compute_vorticity(vel)
    speed = compute_speed(vel)
    ux = vel[0:1]
    uy = vel[1:2]
    field = normalize_field(np.concatenate([ux, uy, speed[None], vort[None]], axis=0))

    canvas_x_start = float(roi["x_min"])
    canvas_x_end = float(roi["x_max"])
    canvas_y_min = float(roi["y_min"])
    canvas_y_max = float(roi["y_max"])

    crops = []
    for scale_name in scales:
        # Parse dist{D}_full
        if scale_name.startswith("dist"):
            parts = scale_name.replace("dist", "").rsplit("_", 1)
            downstream_h = float(parts[0])
            box = build_distance_crop_box(
                downstream_h=downstream_h,
                canvas_x_start=canvas_x_start,
                canvas_x_end=canvas_x_end,
                canvas_y_min=canvas_y_min,
                canvas_y_max=canvas_y_max,
            )
        elif scale_name == "full":
            box = [0.0, 0.0, 1.0, 1.0]
        else:
            box = [0.0, 0.0, 1.0, 1.0]  # fallback

        crop = resize_crop(field, box, output_size)
        crops.append(crop)

    return np.stack(crops, axis=0).astype(np.float32)


# ── Model inference ──────────────────────────────────────────────────────


def predict(x: np.ndarray) -> dict:
    """Run model on [S, 4, H, W] input. Returns prediction dict."""
    if model is None:
        return {"error": "Model not loaded"}

    model.eval()
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)  # [1, S, 4, H, W]
    with torch.no_grad():
        out = model(x_t)

    shape_logits = out["shape_logits"].cpu().numpy()[0]
    params_pred = out["params_pred"].cpu().numpy()[0]
    re_logits = out["re_logits"].cpu().numpy()[0]

    shape_probs = np.exp(shape_logits) / np.exp(shape_logits).sum()
    re_probs = np.exp(re_logits) / np.exp(re_logits).sum()
    shape_idx = int(np.argmax(shape_probs))
    re_idx = int(np.argmax(re_probs))

    return {
        "shape": label_maps.idx_to_shape[shape_idx],
        "shape_probs": {
            label_maps.idx_to_shape[i]: float(shape_probs[i]) for i in range(len(shape_probs))
        },
        "dy": float(params_pred[0]),
        "eps": float(params_pred[1]),
        "re": label_maps.idx_to_re[re_idx],
        "re_probs": {label_maps.idx_to_re[i]: float(re_probs[i]) for i in range(len(re_probs))},
    }


def render_vorticity_fig(
    vort: np.ndarray,
    roi: dict,
    shape: str,
    re_val: float,
    dy: float,
    eps: float,
    pred_shape: str,
    pred_probs: dict,
    pred_dy: float,
    pred_eps: float,
) -> str:
    """Render vorticity field at 3 downstream distances as base64 PNG."""
    canvas_x_start = float(roi["x_min"])
    canvas_x_end = float(roi["x_max"])
    canvas_y_min = float(roi["y_min"])
    canvas_y_max = float(roi["y_max"])

    h, w = vort.shape
    # x positions for each crop
    # dist0.5: starts at x_start + 0.5*h, dist1.0: +1.0*h, dist2.0: +2.0*h
    h_canvas = canvas_y_max - canvas_y_min
    x_starts_phys = [
        canvas_x_start + 0.5 * h_canvas,  # dist0.5
        canvas_x_start + 1.0 * h_canvas,  # dist1.0
        canvas_x_start + 2.0 * h_canvas,  # dist2.0
    ]
    dist_labels = ["0.5h downstream", "1.0h downstream", "2.0h downstream"]

    # Extract crops from the full field
    crops = []
    for x_start_phys in x_starts_phys:
        x0_frac = (x_start_phys - canvas_x_start) / (canvas_x_end - canvas_x_start)
        x0_px = int(np.clip(x0_frac * w, 0, w - 1))
        crops.append(vort[:, x0_px:])

    # Pad to same width
    max_w = max(c.shape[1] for c in crops)
    crops_padded = []
    for c in crops:
        if c.shape[1] < max_w:
            pad = np.zeros((h, max_w - c.shape[1]), dtype=c.dtype)
            crops_padded.append(np.concatenate([c, pad], axis=1))
        else:
            crops_padded.append(c[:, :max_w])

    fig, axes = plt.subplots(
        1, 4, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.65], "wspace": 0.25}
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes[:3]:
        ax.set_facecolor("#1a1a2e")

    vmin, vmax = -4.0, 4.0
    cmap = "RdBu_r"
    extent_y = [canvas_y_min, canvas_y_max]

    for i, (crop, label) in enumerate(zip(crops_padded, dist_labels)):
        ax = axes[i]
        im = ax.imshow(
            crop,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
            extent=[float(x_starts_phys[i]), canvas_x_end, canvas_y_min, canvas_y_max],
        )
        ax.set_title(label, fontsize=9, fontweight="bold", color="#ECEFF1", pad=4)
        ax.tick_params(colors="#78909C", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#37474F")

    # Right: prediction panel
    ax_p = axes[3]
    ax_p.set_facecolor("#1a1a2e")
    ax_p.axis("off")

    shape_color_map = {
        "circle": "#58a6ff",
        "triangle": "#f97583",
        "airfoil": "#d2a8ff",
        "diamond": "#56d364",
        "bar": "#ffa657",
    }
    pred_color = shape_color_map.get(pred_shape, "#90A4AE")
    correct = shape == pred_shape

    ax_p.text(
        0.5,
        0.97,
        "PREDICTION",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#ECEFF1",
        transform=ax_p.transAxes,
    )
    ax_p.text(
        0.5,
        0.88,
        pred_shape.upper(),
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=pred_color,
        transform=ax_p.transAxes,
    )

    badge_bg = "#1B4332" if correct else "#5C1010"
    badge_fc = "#3FB950" if correct else "#F85149"
    badge_txt = "CORRECT" if correct else "WRONG"
    ax_p.add_patch(
        plt.Rectangle(
            (0.05, 0.78),
            0.9,
            0.07,
            facecolor=badge_bg,
            edgecolor=badge_fc,
            lw=1.5,
            transform=ax_p.transAxes,
        )
    )
    ax_p.text(
        0.5,
        0.82,
        badge_txt,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=badge_fc,
        transform=ax_p.transAxes,
    )

    ax_p.text(
        0.5,
        0.74,
        "Confidence",
        ha="center",
        va="top",
        fontsize=8,
        color="#78909C",
        transform=ax_p.transAxes,
    )

    sorted_probs = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)
    for k, (s, p) in enumerate(sorted_probs):
        bar_c = shape_color_map.get(s, "#78909C")
        row_y = 0.68 - k * 0.115
        bold = "bold" if s == pred_shape else "normal"
        ax_p.text(
            0.05,
            row_y,
            s[:7],
            ha="left",
            va="top",
            fontsize=8,
            color=bar_c,
            fontweight=bold,
            transform=ax_p.transAxes,
        )
        ax_p.text(
            0.97,
            row_y,
            f"{p:.0%}",
            ha="right",
            va="top",
            fontsize=8,
            color=bar_c,
            fontweight=bold,
            transform=ax_p.transAxes,
        )
        ax_p.add_patch(
            plt.Rectangle(
                (0.05, row_y - 0.04), 0.9 * p, 0.03, color=bar_c, transform=ax_p.transAxes
            )
        )

    ax_p.text(
        0.5,
        0.12,
        f"dy={pred_dy:.3f}  ε={pred_eps:.3f}",
        ha="center",
        va="top",
        fontsize=8,
        color="#78909C",
        transform=ax_p.transAxes,
    )
    ax_p.text(
        0.5,
        0.05,
        f"Re={re_val}",
        ha="center",
        va="top",
        fontsize=8,
        color="#78909C",
        transform=ax_p.transAxes,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    import base64

    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── Flask routes ─────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    import random as _random

    data = request.get_json() or {}
    shape = str(data.get("shape", "circle"))
    re_val = float(data.get("re", 100))
    dy = float(data.get("dy", 0.0))
    eps = float(data.get("eps", 0.0))
    # Random seed each call → different noise realization for same params
    seed = int(data.get("seed", _random.randint(0, 99999)))
    canvas_size = int(data.get("canvas_size", 256))

    # 1. Generate velocity field
    vel, roi = generate_wake_velocity_field(
        shape=shape, re_val=re_val, dy=dy, eps=eps, seed=seed, canvas_size=canvas_size
    )

    # 2. Compute vorticity for visualization
    vort = compute_vorticity(vel)

    # 3. Prepare model input
    scales = ["dist0.5_full", "dist1.0_full", "dist2.0_full"]
    x = prepare_multiscale_input(vel, roi, scales, canvas_size=canvas_size, output_size=128)

    # 4. Predict
    if model is not None:
        result = predict(x)
        vort_img = render_vorticity_fig(
            vort,
            roi,
            shape,
            re_val,
            dy,
            eps,
            result["shape"],
            result["shape_probs"],
            result["dy"],
            result["eps"],
        )
        result["vorticity_img"] = vort_img
        result["success"] = True
    else:
        result = {
            "success": False,
            "error": "Model not loaded — run training first",
            "shape": "unknown",
            "shape_probs": {},
            "dy": dy,
            "eps": eps,
            "re": re_val,
            "re_probs": {},
        }
        # Still return vorticity image
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        show_x0 = int(vort.shape[1] * 0.05)
        im = ax.imshow(
            vort[:, show_x0:], cmap="RdBu_r", vmin=-4, vmax=4, origin="lower", aspect="auto"
        )
        plt.colorbar(im, ax=ax, label="vorticity")
        ax.set_title(f"No model loaded — showing vorticity (shape={shape})", color="white")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import base64

        result["vorticity_img"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return jsonify(result)


# ── Model loading ─────────────────────────────────────────────────────────


def load_model():
    global model, cfg, label_maps, MODEL_TYPE, MODEL_KWARGS
    cfg = load_config(str(CONFIG_PATH))
    bundle = load_wake_bundle()
    label_maps = build_label_maps(bundle)

    if not MODEL_PATH.exists():
        print(
            f"[WARNING] Model not found at {MODEL_PATH} — running in demo mode (no classification)"
        )
        print(f"[INFO] Run: python -m ml.train_wake --config {CONFIG_PATH}")
        return

    pack = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    MODEL_TYPE = str(pack.get("model_type", "resnet18"))
    MODEL_KWARGS = pack.get("model_kwargs", {})

    if MODEL_TYPE == "mae_vit":
        model = MultiScaleViTWakeNet(**MODEL_KWARGS).to(DEVICE)
    else:
        model = MultiScaleWakeNet(**MODEL_KWARGS).to(DEVICE)
    model.load_state_dict(pack["state_dict"])
    model.eval()
    print(f"[INFO] Model loaded: {MODEL_TYPE} on {DEVICE}")


load_model()

if __name__ == "__main__":
    print("=" * 60)
    print("  Wake-Field Shape Classifier — Interactive Web App")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
