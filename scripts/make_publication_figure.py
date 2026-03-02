from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.config import load_config, repo_root  # noqa: E402
from sim.geometry_mask import render_case_image  # noqa: E402


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a publication-style reproducible figure")
    parser.add_argument("--config", default="configs/exp_180.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output", default="reports/figure_main_reproducible.png", help="Output PNG path"
    )
    parser.add_argument("--dpi", type=int, default=320, help="Output DPI")
    return parser.parse_args()


def _compute_stratified_test_n(n_total: int, n_strata: int, requested_ratio: float) -> int:
    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    return min(max(requested_test_n, min_test_n), max_test_n)


def _get_holdout_indices(strata: np.ndarray, test_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(strata.shape[0])
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_n,
        random_state=seed,
        stratify=strata,
    )
    return np.asarray(idx_train), np.asarray(idx_test)


def _iou_and_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float]:
    yt = y_true >= threshold
    yp = y_pred >= threshold
    inter = np.logical_and(yt, yp).sum()
    union = np.logical_or(yt, yp).sum()
    denom = yt.sum() + yp.sum()

    iou = float(inter / (union + 1e-9))
    dice = float((2.0 * inter) / (denom + 1e-9))
    return iou, dice


def _render_targets(features_df: pd.DataFrame, cfg: dict) -> np.ndarray:
    sim_cfg = cfg["simulation"]
    rec_cfg = cfg["reconstruction"]

    imgs = []
    for _, row in features_df.iterrows():
        img = render_case_image(
            shape=str(row["shape"]),
            dy=float(row["dy"]),
            eps=float(row["eps"]),
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
        imgs.append(img)
    return np.stack(imgs, axis=0)


def _get_commit_short(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    root = repo_root()
    cfg = load_config(args.config)

    features_path = root / "data" / "features" / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run make dataset first.")

    sota_model_path = root / "models" / "sota.pkl"
    recon_model_path = root / "models" / "reconstructor.pkl"
    leaderboard_path = root / "reports" / "model_leaderboard.csv"

    if not sota_model_path.exists():
        raise FileNotFoundError(f"Missing {sota_model_path}. Run make sota first.")
    if not recon_model_path.exists():
        raise FileNotFoundError(f"Missing {recon_model_path}. Run make reconstruct first.")
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing {leaderboard_path}. Run make sota first.")

    features_df = pd.read_csv(features_path)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    x_all = features_df[feature_cols].to_numpy(dtype=float)
    y_shape = features_df["shape"].to_numpy()
    strata = (features_df["shape"].astype(str) + "_Re" + features_df["Re"].astype(str)).to_numpy()

    random_state = int(cfg["ml"].get("random_state", 42))
    test_ratio = float(cfg["ml"].get("test_size", 0.2))
    n_total = int(features_df.shape[0])
    n_strata = int(np.unique(strata).size)
    test_n = _compute_stratified_test_n(n_total=n_total, n_strata=n_strata, requested_ratio=test_ratio)

    idx_train, idx_test = _get_holdout_indices(strata=strata, test_n=test_n, seed=random_state)

    x_test = x_all[idx_test]
    y_test_shape = y_shape[idx_test]

    with sota_model_path.open("rb") as handle:
        sota_pack = pickle.load(handle)
    sota_model = sota_pack["model"]
    labels = sorted(np.unique(y_shape))
    y_pred_shape = sota_model.predict(x_test)

    cm = confusion_matrix(y_test_shape, y_pred_shape, labels=labels)
    cls_acc = float(accuracy_score(y_test_shape, y_pred_shape))
    cls_f1 = float(f1_score(y_test_shape, y_pred_shape, average="macro"))

    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard = leaderboard.sort_values("macro_f1_mean", ascending=True)

    y_target_img = _render_targets(features_df, cfg)
    h_img = int(cfg["reconstruction"]["image_height"])
    w_img = int(cfg["reconstruction"]["image_width"])

    with recon_model_path.open("rb") as handle:
        rec_pack = pickle.load(handle)

    scaler = rec_pack["scaler"]
    pca_y = rec_pack["pca_y"]
    reg = rec_pack["regressor"]
    threshold = float(rec_pack.get("threshold", 0.8))

    y_test_target = y_target_img[idx_test].reshape(len(idx_test), -1)
    z_pred = reg.predict(scaler.transform(x_test))
    y_pred_flat = np.clip(pca_y.inverse_transform(z_pred), 0.0, 1.0)

    mse = float(np.mean((y_test_target - y_pred_flat) ** 2))
    ious = []
    dices = []
    for i in range(y_test_target.shape[0]):
        iou, dice = _iou_and_dice(y_test_target[i], y_pred_flat[i], threshold=threshold)
        ious.append(iou)
        dices.append(dice)
    ious = np.asarray(ious, dtype=float)
    dices = np.asarray(dices, dtype=float)

    eps_test = features_df.iloc[idx_test]["eps"].to_numpy(dtype=float)
    case_ids = features_df.iloc[idx_test]["case_id"].astype(str).to_numpy()

    order = np.argsort(ious)
    example_positions = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
        }
    )

    fig = plt.figure(figsize=(14, 10))
    outer = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)

    ax_a = fig.add_subplot(outer[0, 0])
    im = ax_a.imshow(cm, cmap="Blues")
    ax_a.set_xticks(np.arange(len(labels)), labels=labels, rotation=25)
    ax_a.set_yticks(np.arange(len(labels)), labels=labels)
    ax_a.set_xlabel("Predicted")
    ax_a.set_ylabel("True")
    ax_a.set_title(f"A. Shape Classification (acc={cls_acc:.3f}, F1={cls_f1:.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_a.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax_b = fig.add_subplot(outer[0, 1])
    y_pos = np.arange(leaderboard.shape[0])
    bars = ax_b.barh(
        y_pos,
        leaderboard["macro_f1_mean"],
        xerr=leaderboard["macro_f1_std"],
        color=["#94a3b8"] * (leaderboard.shape[0] - 1) + ["#0ea5e9"],
        alpha=0.95,
        ecolor="#334155",
        capsize=3,
    )
    ax_b.set_yticks(y_pos, labels=leaderboard["model"])
    ax_b.set_xlim(0.0, 1.02)
    ax_b.set_xlabel("Macro F1 (mean ± std over seeds)")
    ax_b.set_title("B. Candidate Model Leaderboard")
    ax_b.grid(axis="x", alpha=0.25)
    for b, v in zip(bars, leaderboard["macro_f1_mean"]):
        ax_b.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    gs_c = outer[1, 0].subgridspec(3, 3, wspace=0.02, hspace=0.08)
    for row, pos in enumerate(example_positions):
        gt = y_test_target[pos].reshape(h_img, w_img)
        pdm = y_pred_flat[pos].reshape(h_img, w_img)
        err = np.abs(gt - pdm)
        title_suffix = f"IoU={ious[pos]:.2f}"

        ax_gt = fig.add_subplot(gs_c[row, 0])
        ax_pd = fig.add_subplot(gs_c[row, 1])
        ax_er = fig.add_subplot(gs_c[row, 2])

        ax_gt.imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        ax_pd.imshow(pdm, cmap="gray", vmin=0.0, vmax=1.0)
        ax_er.imshow(err, cmap="inferno", vmin=0.0, vmax=1.0)

        if row == 0:
            ax_gt.set_title("Target")
            ax_pd.set_title("Reconstruction")
            ax_er.set_title("Abs Error")

        ax_gt.set_ylabel(f"{case_ids[pos]}\n{title_suffix}", fontsize=8)
        for ax in (ax_gt, ax_pd, ax_er):
            ax.set_xticks([])
            ax.set_yticks([])

    dummy_c = fig.add_subplot(outer[1, 0])
    dummy_c.axis("off")
    dummy_c.set_title("C. Geometry Reconstruction: Target vs Predicted", y=1.04)

    ax_d = fig.add_subplot(outer[1, 1])
    eps_abs = np.abs(eps_test)
    ax_d.scatter(eps_abs, ious, alpha=0.72, s=26, color="#1d4ed8", edgecolors="none")

    n_bins = 6
    bins = np.linspace(0.0, max(1e-9, float(np.max(eps_abs))), n_bins + 1)
    centers = []
    mean_iou = []
    std_iou = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (eps_abs >= lo) & (eps_abs <= hi)
        else:
            mask = (eps_abs >= lo) & (eps_abs < hi)
        if np.sum(mask) == 0:
            continue
        centers.append(float((lo + hi) / 2.0))
        mean_iou.append(float(np.mean(ious[mask])))
        std_iou.append(float(np.std(ious[mask], ddof=0)))

    if centers:
        ax_d.errorbar(centers, mean_iou, yerr=std_iou, color="#dc2626", marker="o", lw=2, capsize=3)

    rec_iou_mean = float(np.mean(ious))
    rec_dice_mean = float(np.mean(dices))
    ax_d.set_xlabel("|eps|")
    ax_d.set_ylabel("IoU")
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_title(f"D. Reconstruction Robustness (IoU={rec_iou_mean:.3f}, Dice={rec_dice_mean:.3f})")
    ax_d.grid(alpha=0.25)

    commit_short = _get_commit_short(root)
    fig.suptitle("Reproducible Main Figure: Classification and Geometry Reconstruction", y=0.985)
    fig.text(
        0.01,
        0.006,
        (
            f"Config: {Path(args.config).name} | Cases: {n_total} | Holdout: {test_n} | "
            f"Seed: {random_state} | Commit: {commit_short}"
        ),
        fontsize=9,
        color="#334155",
    )

    output_png = root / args.output
    output_pdf = output_png.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "config": str(args.config),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf),
        "n_cases": n_total,
        "holdout_n": test_n,
        "seed": random_state,
        "commit": commit_short,
        "classification": {
            "accuracy": cls_acc,
            "macro_f1": cls_f1,
        },
        "reconstruction": {
            "mse": mse,
            "iou_mean": rec_iou_mean,
            "dice_mean": rec_dice_mean,
        },
        "example_case_ids": [str(case_ids[i]) for i in example_positions],
    }

    meta_path = output_png.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
