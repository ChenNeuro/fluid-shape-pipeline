"""
Comprehensive wake-field validation visualization.
Saves reports/validation_figure.png with 4 panels:
  A. Example vorticity fields per shape at 3 distance scales
  B. Confusion matrix
  C. dy predictions vs ground truth
  D. Per-shape accuracy bar chart
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.wake_common import build_label_maps, repeated_holdout_split, stratification_labels
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleWakeNet, select_device

# ── config ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "wake_field_smoke.yaml"
MODEL_PATH = REPO_ROOT / "models" / "wake_field_main.pt"
OUTPUT_PATH = REPO_ROOT / "reports" / "validation_figure.png"

cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
smoke_cfg = cfg["vision"]
scales: list[str] = [str(s) for s in smoke_cfg.get("scales", [])]
channels: list[str] = [str(c) for c in smoke_cfg.get("channels", [])]

bundle = load_wake_bundle()
x_all = variant_tensor(bundle, scales=scales, channels=channels)
label_maps = build_label_maps(bundle)
strata = stratification_labels(bundle)

n_total = bundle.case_ids.shape[0]
n_strata = len(np.unique(strata))
test_n = max(n_strata, int(0.2 * n_total))
idx_train, idx_test = repeated_holdout_split(strata, test_n=test_n, seed=42)

x_test = torch.from_numpy(x_all[idx_test]).float()
shapes_test = bundle.shapes[idx_test]
dy_test = bundle.dy[idx_test]
eps_test = bundle.eps[idx_test]

# ── load model ───────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model_kwargs = ckpt["model_kwargs"]
shape_labels_ordered = [str(s) for s in ckpt["shape_labels"]]
n_shapes = len(shape_labels_ordered)
n_scales = len(scales)
n_chans = len(channels)

device = select_device()
model = MultiScaleWakeNet(
    n_scales=n_scales,
    in_channels=n_chans,
    n_shapes=n_shapes,
    n_re_classes=int(model_kwargs["n_re_classes"]),
    fusion_hidden=int(smoke_cfg.get("training", {}).get("fusion_hidden", 128)),
    dropout=float(smoke_cfg.get("training", {}).get("dropout", 0.1)),
).to(device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

# ── inference ───────────────────────────────────────────────────────────────
with torch.no_grad():
    outputs = model(x_test.to(device))
    shape_logits = outputs["shape_logits"].cpu()
    params_pred = outputs["params_pred"].cpu()

shape_pred_idx = shape_logits.argmax(dim=1).numpy()
shape_pred = np.array([shape_labels_ordered[i] for i in shape_pred_idx])
shape_idx_test = np.array([shape_labels_ordered.index(str(s)) for s in shapes_test])

pert_min = float(cfg["simulation"]["perturb"]["dy_min"])
pert_max = float(cfg["simulation"]["perturb"]["dy_max"])
eps_max = float(cfg["simulation"]["perturb"]["eps_max"])
dy_pred = torch.tanh(params_pred[:, 0]).numpy() * pert_max
eps_pred = torch.tanh(params_pred[:, 1]).numpy() * eps_max

accuracy = float(np.mean(shape_pred_idx == shape_idx_test))
dy_mae = float(np.mean(np.abs(dy_pred - dy_test)))
eps_mae = float(np.mean(np.abs(eps_pred - eps_test)))
macro_f1 = float(
    np.mean(
        [
            np.mean(shape_pred_idx[shape_idx_test == i] == i)
            for i in range(n_shapes)
            if np.any(shape_idx_test == i)
        ]
    )
)
print(
    f"Accuracy: {accuracy:.1%} | Macro-F1: {macro_f1:.1%} | dy_MAE: {dy_mae:.5f} | eps_MAE: {eps_mae:.5f}"
)


# ── colormaps ───────────────────────────────────────────────────────────────
def vorticity_cmap():
    colors = [
        "#08306b",
        "#2171b5",
        "#6baed6",
        "#c6dbef",
        "#f7fbff",
        "#fff5f0",
        "#fee0d2",
        "#fc9272",
        "#de2d26",
        "#67000d",
    ]
    return LinearSegmentedColormap.from_list("vort", colors, N=256)


CHANNEL_CMAPS = [plt.cm.RdBu_r, plt.cm.RdBu_r, plt.cm.YlOrRd, vorticity_cmap()]
CHANNEL_DISPLAY = ["uₓ", "uᵧ", "Speed", "Vorticity"]
SHAPE_COLORS = {
    "circle": "#60a5fa",
    "triangle": "#f97316",
    "airfoil": "#a78bfa",
    "diamond": "#34d399",
    "bar": "#fbbf24",
}
SCALE_LABELS = {"dist0.5_full": "0.5h", "dist1.0_full": "1.0h", "dist2.0_full": "2.0h"}

# ── figure layout ──────────────────────────────────────────────────────────
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
TEXT_SEC = "#8b949e"

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor(BG)

outer = gridspec.GridSpec(
    2,
    4,
    figure=fig,
    hspace=0.42,
    wspace=0.30,
    left=0.05,
    right=0.97,
    top=0.90,
    bottom=0.10,
    width_ratios=[1, 1, 1, 1],
)

# Panel A: nested grid for wake fields (occupies cols 0-2 of row 0)
panel_a_gs = gridspec.GridSpecFromSubplotSpec(
    n_shapes,
    n_scales,
    subplot_spec=outer[0, :3],
    hspace=0.18,
    wspace=0.10,
)

# Panel B: confusion matrix
ax_b = fig.add_subplot(outer[0, 3])

# Panels C/D/E: bottom row
ax_c = fig.add_subplot(outer[1, 0])  # dy scatter
ax_d = fig.add_subplot(outer[1, 1])  # eps scatter
ax_e = fig.add_subplot(outer[1, 2])  # per-shape accuracy
ax_legend = fig.add_subplot(outer[1, 3])  # shape legend + Re breakdown
ax_legend.set_facecolor(BG)
ax_legend.axis("off")

# ── Panel A: Wake field examples ───────────────────────────────────────────
# Find best example per shape (closest to median dy)
example_idx = {}
for shape in shape_labels_ordered:
    mask = shapes_test == shape
    if mask.any():
        candidates = np.where(mask)[0]
        dy_cands = dy_test[candidates]
        example_idx[shape] = candidates[np.argmin(np.abs(dy_cands - np.median(dy_cands)))]

for shi, shape in enumerate(shape_labels_ordered):
    if shape not in example_idx:
        continue
    local_idx = example_idx[shape]
    for si, scale in enumerate(scales):
        # Only show vorticity channel (most informative)
        ci_vort = channels.index("vorticity")
        ax = fig.add_subplot(panel_a_gs[shi, si])
        img = bundle.crops_by_scale[scale][idx_test[local_idx], ci_vort]
        vabs = np.percentile(np.abs(img), 98) + 1e-8
        ax.imshow(
            img, cmap=CHANNEL_CMAPS[ci_vort], vmin=-vabs, vmax=vabs, origin="lower", aspect="auto"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_facecolor(BG)

        # Labels
        if shi == 0:
            ax.set_title(
                SCALE_LABELS.get(scale, scale), color=FG, fontsize=11, fontweight="bold", pad=4
            )
        if si == 0:
            ax.set_ylabel(
                shape.capitalize(),
                color=SHAPE_COLORS.get(shape, FG),
                fontsize=11,
                fontweight="bold",
                rotation=90,
                labelpad=6,
            )
        if shi == n_shapes - 1 and si == n_scales // 2:
            ax.set_xlabel("Downstream distance →", color=TEXT_SEC, fontsize=10)

fig.text(
    0.26,
    0.97,
    "A. Vorticity Fields — 3 Downstream Distances",
    color=FG,
    fontsize=13,
    fontweight="bold",
    ha="center",
    va="top",
)

# ── Panel B: Confusion Matrix ───────────────────────────────────────────────
ax_b.set_facecolor("#161b22")
cm = confusion_matrix(shape_idx_test, shape_pred_idx, labels=list(range(n_shapes)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
im = ax_b.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
ax_b.set_xticks(range(n_shapes))
ax_b.set_yticks(range(n_shapes))
labels = [s.capitalize() for s in shape_labels_ordered]
ax_b.set_xticklabels(labels, color=TEXT_SEC, fontsize=9, rotation=35, ha="right")
ax_b.set_yticklabels(labels, color=TEXT_SEC, fontsize=9)
ax_b.set_xlabel("Predicted", color=FG, fontsize=10)
ax_b.set_ylabel("True", color=FG, fontsize=10)
ax_b.set_title(
    f"B. Confusion Matrix\nAcc={accuracy:.0%}  F1={macro_f1:.0%}",
    color=FG,
    fontsize=11,
    fontweight="bold",
)
for i in range(n_shapes):
    for j in range(n_shapes):
        color = "#fff" if cm_norm[i, j] < 0.5 else "#000"
        ax_b.text(
            j,
            i,
            f"{cm_norm[i, j]:.0%}",
            ha="center",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )
ax_b.spines[:].set_color(GRID)
plt.colorbar(im, ax=ax_b, label="Recall", pad=0.02)

# ── Panel C: dy scatter ────────────────────────────────────────────────────
ax_c.set_facecolor("#161b22")
for shape in shape_labels_ordered:
    mask = shapes_test == shape
    ax_c.scatter(
        dy_test[mask],
        dy_pred[mask],
        color=SHAPE_COLORS.get(shape, FG),
        alpha=0.75,
        s=60,
        edgecolors="none",
    )
diag = np.linspace(pert_min, pert_max, 200)
ax_c.plot(diag, diag, color="#f85149", linewidth=1.8, linestyle="--", label="Perfect", zorder=0)
ax_c.set_xlabel("True dy", color=FG, fontsize=10)
ax_c.set_ylabel("Predicted dy", color=FG, fontsize=10)
ax_c.set_title(f"C. dy Prediction\nMAE={dy_mae:.4f}", color=FG, fontsize=11, fontweight="bold")
ax_c.set_xlim(pert_min - 0.01, pert_max + 0.01)
ax_c.set_ylim(pert_min - 0.01, pert_max + 0.01)
ax_c.tick_params(colors=TEXT_SEC)
for spine in ax_c.spines.values():
    spine.set_color(GRID)
ax_c.grid(True, alpha=0.12, color=GRID)
ax_c.plot([], [], color="#f85149", linewidth=1.5, linestyle="--", label="Perfect")
ax_c.legend(fontsize=8, loc="upper left", framealpha=0.3, labelcolor=FG)

# ── Panel D: eps scatter ───────────────────────────────────────────────────
ax_d.set_facecolor("#161b22")
for shape in shape_labels_ordered:
    mask = shapes_test == shape
    ax_d.scatter(
        eps_test[mask],
        eps_pred[mask],
        color=SHAPE_COLORS.get(shape, FG),
        alpha=0.75,
        s=60,
        edgecolors="none",
    )
diag_eps = np.linspace(-eps_max, eps_max, 200)
ax_d.plot(
    diag_eps, diag_eps, color="#f85149", linewidth=1.8, linestyle="--", label="Perfect", zorder=0
)
ax_d.set_xlabel("True ε", color=FG, fontsize=10)
ax_d.set_ylabel("Predicted ε", color=FG, fontsize=10)
ax_d.set_title(f"D. ε Prediction\nMAE={eps_mae:.4f}", color=FG, fontsize=11, fontweight="bold")
ax_d.set_xlim(-eps_max - 0.01, eps_max + 0.01)
ax_d.set_ylim(-eps_max - 0.01, eps_max + 0.01)
ax_d.tick_params(colors=TEXT_SEC)
for spine in ax_d.spines.values():
    spine.set_color(GRID)
ax_d.grid(True, alpha=0.12, color=GRID)
ax_d.plot([], [], color="#f85149", linewidth=1.5, linestyle="--", label="Perfect")
ax_d.legend(fontsize=8, loc="upper left", framealpha=0.3, labelcolor=FG)

# ── Panel E: Per-shape accuracy ────────────────────────────────────────────
shape_acc = {}
for shape in shape_labels_ordered:
    mask = shapes_test == shape
    if mask.any():
        shape_acc[shape] = float(np.mean(shape_pred[mask] == shapes_test[mask]))
    else:
        shape_acc[shape] = 0.0

ax_e.set_facecolor("#161b22")
names = [s.capitalize() for s in shape_labels_ordered]
values = [shape_acc[s] for s in shape_labels_ordered]
colors_bar = [SHAPE_COLORS.get(s, FG) for s in shape_labels_ordered]
bars = ax_e.bar(names, values, color=colors_bar, edgecolor="none", alpha=0.88)
for bar, val in zip(bars, values):
    ax_e.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.02,
        f"{val:.0%}",
        ha="center",
        va="bottom",
        color=FG,
        fontsize=10,
        fontweight="bold",
    )
ax_e.axhline(accuracy, color="#f85149", linewidth=1.5, linestyle="--")
ax_e.text(
    n_shapes - 0.5,
    accuracy + 0.03,
    f"Overall\n{accuracy:.0%}",
    color="#f85149",
    fontsize=8,
    ha="center",
    va="bottom",
)
ax_e.set_ylim(0, 1.18)
ax_e.set_ylabel("Accuracy", color=FG, fontsize=10)
ax_e.set_title("E. Per-Shape Accuracy", color=FG, fontsize=11, fontweight="bold")
ax_e.tick_params(colors=TEXT_SEC)
ax_e.set_xticklabels(names, rotation=25, ha="right", color=TEXT_SEC, fontsize=9)
for spine in ax_e.spines.values():
    spine.set_color(GRID)
ax_e.grid(True, alpha=0.12, axis="y", color=GRID)

# ── Panel F: Legend + Re breakdown ─────────────────────────────────────────
re_acc = {}
for re_val in sorted(np.unique(bundle.re_values[idx_test])):
    mask = bundle.re_values[idx_test] == re_val
    re_acc[int(re_val)] = float(np.mean(shape_pred_idx[mask] == shape_idx_test[mask]))

ax_legend.set_facecolor("#161b22")
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.axis("off")

y = 0.95
ax_legend.text(
    0.5, y, "Shape Legend", ha="center", va="top", color=FG, fontsize=11, fontweight="bold"
)
y -= 0.10
for shape in shape_labels_ordered:
    color = SHAPE_COLORS.get(shape, FG)
    ax_legend.text(
        0.05, y, "●", color=color, fontsize=14, va="center", transform=ax_legend.transAxes
    )
    ax_legend.text(
        0.15,
        y,
        shape.capitalize(),
        color=FG,
        fontsize=10,
        va="center",
        transform=ax_legend.transAxes,
    )
    y -= 0.12

y -= 0.08
ax_legend.text(
    0.5, y, "Per-Re Accuracy", ha="center", va="top", color=FG, fontsize=11, fontweight="bold"
)
y -= 0.10
for re_val, acc in re_acc.items():
    ax_legend.text(
        0.05,
        y,
        f"Re={re_val}:",
        color=TEXT_SEC,
        fontsize=10,
        va="center",
        transform=ax_legend.transAxes,
    )
    ax_legend.text(
        0.55,
        y,
        f"{acc:.0%}",
        color=FG,
        fontsize=10,
        fontweight="bold",
        va="center",
        transform=ax_legend.transAxes,
    )
    y -= 0.12

# ── global title ───────────────────────────────────────────────────────────
test_n_total = len(idx_test)
fig.suptitle(
    f"Wake-Field Validation  |  dist_multi_4ch (0.5h / 1.0h / 2.0h downstream, 4 channels)"
    f"  |  N={test_n_total} test cases  |  "
    f"Acc={accuracy:.1%}  F1={macro_f1:.1%}  "
    f"dy_MAE={dy_mae:.4f}  ε_MAE={eps_mae:.4f}",
    color=FG,
    fontsize=12,
    fontweight="bold",
    y=0.98,
)

# ── save ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PATH}")
