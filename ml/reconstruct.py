from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sim.config import load_config, repo_root
from sim.geometry_mask import render_case_image
from sim.logging_utils import setup_logger


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]
RECON_METHODS = {"latent_ridge", "parametric_inverse"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct geometry images from probe features")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def _compute_stratified_test_n(n_total: int, n_strata: int, requested_ratio: float) -> int:
    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    return min(max(requested_test_n, min_test_n), max_test_n)


def _render_targets(features_df: pd.DataFrame, cfg: dict, image_height: int, image_width: int) -> np.ndarray:
    sim_cfg = cfg["simulation"]
    images = []
    for _, row in features_df.iterrows():
        images.append(
            render_case_image(
                shape=str(row["shape"]),
                dy=float(row["dy"]),
                eps=float(row["eps"]),
                h=float(sim_cfg["H"]),
                d_ratio=float(sim_cfg["d_ratio"]),
                x0=float(sim_cfg["x0"]),
                y0=float(sim_cfg["y0"]),
                l_in=float(sim_cfg["L_in"]),
                l_out=float(sim_cfg["L_out"]),
                image_height=image_height,
                image_width=image_width,
                eps_max_for_canvas=float(sim_cfg["perturb"]["eps_max"]),
            )
        )
    return np.stack(images, axis=0)


def _prepare_xy(features_df: pd.DataFrame, cfg: dict):
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    x = features_df[feature_cols].to_numpy(dtype=float)
    y_shape = features_df["shape"].astype(str).to_numpy()
    y_params = features_df[["dy", "eps"]].to_numpy(dtype=float)

    rec_cfg = cfg["reconstruction"]
    h_img = int(rec_cfg["image_height"])
    w_img = int(rec_cfg["image_width"])
    y_img = _render_targets(features_df, cfg, image_height=h_img, image_width=w_img)
    y_flat = y_img.reshape(y_img.shape[0], -1)

    strata = (features_df["shape"].astype(str) + "_Re" + features_df["Re"].astype(str)).to_numpy()
    return x, y_shape, y_params, y_img, y_flat, strata, feature_cols


def _iou_and_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float]:
    yt = y_true >= threshold
    yp = y_pred >= threshold
    inter = np.logical_and(yt, yp).sum()
    union = np.logical_or(yt, yp).sum()
    dice_den = yt.sum() + yp.sum()

    iou = float(inter / (union + 1e-9))
    dice = float((2.0 * inter) / (dice_den + 1e-9))
    return iou, dice


def _geometry_classes(flat_img: np.ndarray) -> np.ndarray:
    cls = np.zeros(flat_img.shape, dtype=np.int8)
    cls[(flat_img >= 0.2) & (flat_img < 0.8)] = 1
    cls[flat_img >= 0.8] = 2
    return cls


def _multiclass_iou_from_class_maps(cls_true: np.ndarray, cls_pred: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    ious: list[float] = []
    for cls in (0, 1, 2):
        inter = np.logical_and(cls_true == cls, cls_pred == cls).sum()
        union = np.logical_or(cls_true == cls, cls_pred == cls).sum()
        ious.append(float(inter / (union + 1e-9)))
    return float(np.mean(ious)), (ious[0], ious[1], ious[2])


def _evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    iou_list = []
    dice_list = []
    miou3_list = []
    miou3_fluid = []
    miou3_wall = []
    miou3_obstacle = []
    for i in range(y_true.shape[0]):
        iou, dice = _iou_and_dice(y_true[i], y_pred[i], threshold=threshold)
        iou_list.append(iou)
        dice_list.append(dice)
        cls_true = _geometry_classes(y_true[i])
        cls_pred = _geometry_classes(y_pred[i])
        miou3, (iou_fluid, iou_wall, iou_obstacle) = _multiclass_iou_from_class_maps(cls_true, cls_pred)
        miou3_list.append(miou3)
        miou3_fluid.append(iou_fluid)
        miou3_wall.append(iou_wall)
        miou3_obstacle.append(iou_obstacle)
    return {
        "mse": mse,
        "iou": float(np.mean(iou_list)),
        "dice": float(np.mean(dice_list)),
        "miou3": float(np.mean(miou3_list)),
        "miou3_fluid": float(np.mean(miou3_fluid)),
        "miou3_wall": float(np.mean(miou3_wall)),
        "miou3_obstacle": float(np.mean(miou3_obstacle)),
        "iou_values": np.asarray(iou_list, dtype=float),
        "dice_values": np.asarray(dice_list, dtype=float),
        "miou3_values": np.asarray(miou3_list, dtype=float),
        "miou3_fluid_values": np.asarray(miou3_fluid, dtype=float),
        "miou3_wall_values": np.asarray(miou3_wall, dtype=float),
        "miou3_obstacle_values": np.asarray(miou3_obstacle, dtype=float),
    }


def _clip_params(dy: np.ndarray, eps: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pert_cfg = cfg["simulation"]["perturb"]
    if bool(pert_cfg.get("enable_dy", True)):
        dy_min = float(pert_cfg.get("dy_min", -0.02))
        dy_max = float(pert_cfg.get("dy_max", 0.02))
        dy = np.clip(dy, dy_min, dy_max)
    else:
        dy = np.zeros_like(dy)

    if bool(pert_cfg.get("enable_eps", True)):
        eps_max = float(pert_cfg.get("eps_max", 0.02))
        eps = np.clip(eps, -eps_max, eps_max)
    else:
        eps = np.zeros_like(eps)
    return dy, eps


def _fit_latent_ridge(
    x_train: np.ndarray,
    y_train_flat: np.ndarray,
    latent_dim: int,
    ridge_alpha: float,
    seed: int,
) -> dict:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    n_components = max(2, min(latent_dim, y_train_flat.shape[0] - 1, y_train_flat.shape[1]))
    pca_y = PCA(n_components=n_components, random_state=int(seed))
    z_train = pca_y.fit_transform(y_train_flat)

    reg = Ridge(alpha=ridge_alpha)
    reg.fit(x_train_s, z_train)

    return {"scaler": scaler, "pca_y": pca_y, "regressor": reg}


def _predict_latent_ridge(model_pack: dict, x_test: np.ndarray) -> tuple[np.ndarray, dict]:
    z_pred = model_pack["regressor"].predict(model_pack["scaler"].transform(x_test))
    y_pred = np.clip(model_pack["pca_y"].inverse_transform(z_pred), 0.0, 1.0)
    return y_pred, {}


def _fit_parametric_inverse(
    x_train: np.ndarray,
    y_shape_train: np.ndarray,
    y_params_train: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    rec_cfg = cfg["reconstruction"]
    ml_cfg = cfg["ml"]
    n_estimators = int(rec_cfg.get("param_n_estimators", max(600, int(ml_cfg.get("n_estimators", 400)) * 2)))
    min_samples_leaf = int(rec_cfg.get("param_min_samples_leaf", 1))

    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=int(seed),
        n_jobs=-1,
        class_weight="balanced",
    )
    regressor = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=int(seed),
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
    )

    classifier.fit(x_train, y_shape_train)
    regressor.fit(x_train, y_params_train)
    return {"classifier": classifier, "param_regressor": regressor}


def _render_from_predicted_params(
    shape_pred: np.ndarray,
    dy_pred: np.ndarray,
    eps_pred: np.ndarray,
    cfg: dict,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    sim_cfg = cfg["simulation"]
    imgs = []
    for shape, dy, eps in zip(shape_pred, dy_pred, eps_pred):
        imgs.append(
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
                image_height=image_height,
                image_width=image_width,
                eps_max_for_canvas=float(sim_cfg["perturb"]["eps_max"]),
            )
        )
    return np.stack(imgs, axis=0)


def _predict_parametric_inverse(
    model_pack: dict,
    x_test: np.ndarray,
    cfg: dict,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, dict]:
    classifier = model_pack["classifier"]
    regressor = model_pack["param_regressor"]

    shape_pred = classifier.predict(x_test)
    params_pred = regressor.predict(x_test)
    if params_pred.ndim == 1:
        params_pred = params_pred.reshape(-1, 2)
    dy_pred, eps_pred = _clip_params(params_pred[:, 0], params_pred[:, 1], cfg)

    shape_confidence = None
    if hasattr(classifier, "predict_proba"):
        shape_confidence = np.max(classifier.predict_proba(x_test), axis=1)

    dy_std = None
    eps_std = None
    estimators = getattr(regressor, "estimators_", None)
    if estimators:
        per_tree = np.stack([est.predict(x_test) for est in estimators], axis=0)
        if per_tree.ndim == 2:
            per_tree = per_tree[:, :, None]
        if per_tree.shape[2] >= 2:
            dy_std = per_tree[:, :, 0].std(axis=0)
            eps_std = per_tree[:, :, 1].std(axis=0)

    pred_img = _render_from_predicted_params(
        shape_pred=shape_pred,
        dy_pred=dy_pred,
        eps_pred=eps_pred,
        cfg=cfg,
        image_height=image_height,
        image_width=image_width,
    )
    return pred_img.reshape(pred_img.shape[0], -1), {
        "shape_pred": np.asarray(shape_pred, dtype=str),
        "dy_pred": np.asarray(dy_pred, dtype=float),
        "eps_pred": np.asarray(eps_pred, dtype=float),
        "shape_confidence": np.asarray(shape_confidence, dtype=float) if shape_confidence is not None else None,
        "dy_std": np.asarray(dy_std, dtype=float) if dy_std is not None else None,
        "eps_std": np.asarray(eps_std, dtype=float) if eps_std is not None else None,
    }


def fit_reconstruction_method(
    method: str,
    x_train: np.ndarray,
    y_shape_train: np.ndarray,
    y_params_train: np.ndarray,
    y_train_flat: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    rec_cfg = cfg["reconstruction"]
    if method == "latent_ridge":
        return _fit_latent_ridge(
            x_train=x_train,
            y_train_flat=y_train_flat,
            latent_dim=int(rec_cfg.get("latent_dim", 64)),
            ridge_alpha=float(rec_cfg.get("ridge_alpha", 1.0)),
            seed=seed,
        )
    if method == "parametric_inverse":
        return _fit_parametric_inverse(
            x_train=x_train,
            y_shape_train=y_shape_train,
            y_params_train=y_params_train,
            cfg=cfg,
            seed=seed,
        )
    raise ValueError(f"Unsupported reconstruction method: {method}")


def predict_reconstruction_images(
    method: str,
    fitted_model: dict,
    x_test: np.ndarray,
    cfg: dict,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, dict]:
    if method == "latent_ridge":
        return _predict_latent_ridge(model_pack=fitted_model, x_test=x_test)
    if method == "parametric_inverse":
        return _predict_parametric_inverse(
            model_pack=fitted_model,
            x_test=x_test,
            cfg=cfg,
            image_height=image_height,
            image_width=image_width,
        )
    raise ValueError(f"Unsupported reconstruction method: {method}")


def _plot_examples(
    y_true_img: np.ndarray,
    y_pred_img: np.ndarray,
    case_ids: list[str],
    output_path: Path,
    method_name: str,
    pred_shape: list[str] | None = None,
    n_show: int = 6,
) -> None:
    n = min(n_show, y_true_img.shape[0])
    idx = np.linspace(0, y_true_img.shape[0] - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(10, 2.35 * n))
    if n == 1:
        axes = np.array([axes])

    for row, i in enumerate(idx):
        gt = y_true_img[i]
        pdm = y_pred_img[i]
        err = np.abs(gt - pdm)

        pred_txt = f" pred={pred_shape[i]}" if pred_shape is not None else ""
        axes[row, 0].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"GT {case_ids[i]}")
        axes[row, 1].imshow(pdm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(f"Pred{pred_txt}")
        axes[row, 2].imshow(err, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("|Err|")
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(f"Reconstruction Examples ({method_name})", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_iou_vs_eps(eps: np.ndarray, iou: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(np.abs(eps), iou, alpha=0.85)
    if iou.size >= 2:
        order = np.argsort(np.abs(eps))
        xs = np.abs(eps)[order]
        ys = iou[order]
        ax.plot(xs, ys, alpha=0.5)
    ax.set_xlabel("|eps|")
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Reconstruction IoU vs Geometry Perturbation")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confidence_vs_iou(shape_confidence: np.ndarray, iou_values: np.ndarray, output_path: Path) -> None:
    if shape_confidence.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(shape_confidence, iou_values, alpha=0.85)
    if shape_confidence.size > 1:
        order = np.argsort(shape_confidence)
        xs = shape_confidence[order]
        ys = iou_values[order]
        ax.plot(xs, ys, alpha=0.45)
    ax.set_xlabel("Shape confidence")
    ax.set_ylabel("Obstacle IoU")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Reconstruction Quality vs Shape Confidence")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_case_metrics_df(
    features_df: pd.DataFrame,
    idx_test: np.ndarray,
    y_shape_test: np.ndarray,
    y_params_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, np.ndarray],
    aux: dict,
) -> pd.DataFrame:
    case_df = pd.DataFrame(
        {
            "case_id": features_df.iloc[idx_test]["case_id"].astype(str).to_numpy(),
            "shape_true": y_shape_test.astype(str),
            "Re": features_df.iloc[idx_test]["Re"].astype(int).to_numpy(),
            "dy_true": y_params_test[:, 0].astype(float),
            "eps_true": y_params_test[:, 1].astype(float),
            "iou_obstacle": metrics["iou_values"].astype(float),
            "dice_obstacle": metrics["dice_values"].astype(float),
            "miou3": metrics["miou3_values"].astype(float),
            "miou3_fluid": metrics["miou3_fluid_values"].astype(float),
            "miou3_wall": metrics["miou3_wall_values"].astype(float),
            "miou3_obstacle": metrics["miou3_obstacle_values"].astype(float),
            "mse_case": np.mean((y_true - y_pred) ** 2, axis=1).astype(float),
        }
    )

    if aux.get("shape_pred") is not None:
        case_df["shape_pred"] = aux["shape_pred"].astype(str)
        case_df["shape_correct"] = (case_df["shape_true"] == case_df["shape_pred"]).astype(int)
    if aux.get("shape_confidence") is not None:
        case_df["shape_confidence"] = aux["shape_confidence"].astype(float)
    if aux.get("dy_pred") is not None:
        case_df["dy_pred"] = aux["dy_pred"].astype(float)
        case_df["dy_abs_err"] = np.abs(case_df["dy_true"] - case_df["dy_pred"])
    if aux.get("eps_pred") is not None:
        case_df["eps_pred"] = aux["eps_pred"].astype(float)
        case_df["eps_abs_err"] = np.abs(case_df["eps_true"] - case_df["eps_pred"])
        case_df["eps_sign_correct"] = ((case_df["eps_true"] >= 0) == (case_df["eps_pred"] >= 0)).astype(int)
    if aux.get("dy_std") is not None:
        case_df["dy_std"] = aux["dy_std"].astype(float)
    if aux.get("eps_std") is not None:
        case_df["eps_std"] = aux["eps_std"].astype(float)

    return case_df.sort_values(["iou_obstacle", "mse_case"], ascending=[True, False]).reset_index(drop=True)


def _random_pair_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, seed: int = 123) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_true.shape[0])
    return _evaluate_prediction(y_true=y_true[perm], y_pred=y_pred, threshold=threshold)


def _write_sanity_report(
    output_path: Path,
    aligned_metrics: dict[str, float],
    random_metrics: dict[str, float],
    case_metrics_df: pd.DataFrame,
    confidence_threshold: float | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Reconstruction Sanity Report")
    lines.append("")
    lines.append("## Aligned vs Random-Pair Control")
    lines.append(
        f"- Obstacle IoU: aligned={aligned_metrics['iou']:.4f}, random_pair={random_metrics['iou']:.4f}, "
        f"delta={aligned_metrics['iou'] - random_metrics['iou']:.4f}"
    )
    lines.append(
        f"- Obstacle Dice: aligned={aligned_metrics['dice']:.4f}, random_pair={random_metrics['dice']:.4f}, "
        f"delta={aligned_metrics['dice'] - random_metrics['dice']:.4f}"
    )
    lines.append(
        f"- Geometry mIoU(3-class): aligned={aligned_metrics['miou3']:.4f}, random_pair={random_metrics['miou3']:.4f}, "
        f"delta={aligned_metrics['miou3'] - random_metrics['miou3']:.4f}"
    )
    lines.append("")

    lines.append("## Low-Quality Tail")
    n = int(case_metrics_df.shape[0])
    low05 = int((case_metrics_df["iou_obstacle"] < 0.5).sum())
    low07 = int((case_metrics_df["iou_obstacle"] < 0.7).sum())
    lines.append(f"- Cases with obstacle IoU < 0.50: {low05}/{n} ({(low05 / max(n, 1)) * 100.0:.2f}%)")
    lines.append(f"- Cases with obstacle IoU < 0.70: {low07}/{n} ({(low07 / max(n, 1)) * 100.0:.2f}%)")

    if "shape_correct" in case_metrics_df.columns:
        lines.append(f"- Shape accuracy on this holdout: {case_metrics_df['shape_correct'].mean():.4f}")
    if "shape_confidence" in case_metrics_df.columns:
        corr = np.corrcoef(case_metrics_df["shape_confidence"].to_numpy(), case_metrics_df["iou_obstacle"].to_numpy())[0, 1]
        lines.append(f"- Corr(shape_confidence, obstacle_iou): {float(corr):.4f}")
        if confidence_threshold is not None:
            flagged = case_metrics_df["shape_confidence"] < float(confidence_threshold)
            n_flagged = int(flagged.sum())
            bad = case_metrics_df["iou_obstacle"] < 0.7
            bad_in_flagged = int((flagged & bad).sum())
            lines.append(
                f"- Low-confidence flag (<{confidence_threshold:.2f}): {n_flagged}/{n} flagged; "
                f"{bad_in_flagged}/{max(n_flagged,1)} flagged cases have IoU<0.7"
            )
    lines.append("")

    lines.append("## Worst Cases (Top 10 by lowest obstacle IoU)")
    cols = ["case_id", "shape_true", "shape_pred", "Re", "iou_obstacle", "miou3"]
    avail_cols = [c for c in cols if c in case_metrics_df.columns]
    for _, row in case_metrics_df.head(10)[avail_cols].iterrows():
        lines.append(
            "- "
            + ", ".join(
                [f"{c}={row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else f"{c}={row[c]}" for c in avail_cols]
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_method_comparison(method_summary_df: pd.DataFrame, output_path: Path) -> None:
    if method_summary_df.empty:
        return

    data = method_summary_df.sort_values("iou_mean", ascending=True)
    y = np.arange(data.shape[0], dtype=float)
    labels = data["method"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(
        y,
        data["iou_mean"],
        xerr=data["iou_std"],
        color="#0ea5e9",
        alpha=0.9,
        ecolor="#334155",
        capsize=3,
    )
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("IoU mean ± std (repeated holdout)")
    ax.set_title("Reconstruction Method Comparison")
    ax.grid(axis="x", alpha=0.25)
    for bar, mean_iou in zip(bars, data["iou_mean"]):
        ax.text(mean_iou + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{mean_iou:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _summarize_by_method(repeat_df: pd.DataFrame) -> pd.DataFrame:
    if repeat_df.empty:
        raise RuntimeError("No reconstruction evaluation rows were generated")
    summary = (
        repeat_df.groupby("method", as_index=False)
        .agg(
            n_repeats=("seed", "count"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            miou3_mean=("miou3", "mean"),
            miou3_std=("miou3", "std"),
            shape_acc_mean=("shape_acc", "mean"),
            dy_mae_mean=("dy_mae", "mean"),
            eps_mae_mean=("eps_mae", "mean"),
        )
        .fillna(0.0)
    )
    summary = summary.sort_values(["miou3_mean", "iou_mean", "dice_mean", "mse_mean"], ascending=[False, False, False, True]).reset_index(drop=True)
    summary["method_rank"] = np.arange(1, summary.shape[0] + 1)
    return summary


def _write_summary(
    output_path: Path,
    selected_method: str,
    method_summary_df: pd.DataFrame,
    test_n: int,
    train_n: int,
    repeat_seeds: list[int],
    re_rows: list[dict[str, float]],
) -> None:
    lines = []
    lines.append("# Geometry Reconstruction Summary")
    lines.append("")
    lines.append("## Split Setup")
    lines.append(f"- Stratified by (shape, Re), train={train_n}, test={test_n}")
    lines.append(f"- Repeated seeds: {repeat_seeds}")
    lines.append(f"- Selected method: `{selected_method}`")
    lines.append("")
    lines.append("## Method Leaderboard (Repeated Holdout)")
    for _, row in method_summary_df.iterrows():
        lines.append(
            "- "
            + (
                f"{row['method']}: IoU={row['iou_mean']:.4f}±{row['iou_std']:.4f}, "
                f"Dice={row['dice_mean']:.4f}±{row['dice_std']:.4f}, "
                f"mIoU3={row['miou3_mean']:.4f}±{row['miou3_std']:.4f}, "
                f"MSE={row['mse_mean']:.6f}±{row['mse_std']:.6f}, "
                f"shape_acc={row['shape_acc_mean']:.4f}, "
                f"dy_MAE={row['dy_mae_mean']:.5f}, eps_MAE={row['eps_mae_mean']:.5f}"
            )
        )
    lines.append("")
    lines.append("## Leave-One-Re-Out (Selected Method)")
    for row in re_rows:
        lines.append(
            f"- Re={row['Re_test']} (n={row['n_test']}): MSE={row['mse']:.6f}, IoU={row['iou']:.4f}, "
            f"Dice={row['dice']:.4f}, mIoU3={row['miou3']:.4f}, shape_acc={row['shape_acc']:.4f}, "
            f"dy_MAE={row['dy_mae']:.5f}, eps_MAE={row['eps_mae']:.5f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("reconstruct", root / "logs" / "reconstruct.log")

    features_path = root / "data" / "features" / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run make dataset first.")

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        raise RuntimeError("features.csv is empty")

    x, y_shape, y_params, y_img, y_flat, strata, feature_cols = _prepare_xy(features_df, cfg)

    rec_cfg = cfg["reconstruction"]
    h_img = int(rec_cfg["image_height"])
    w_img = int(rec_cfg["image_width"])
    obstacle_threshold = float(rec_cfg.get("obstacle_threshold", 0.8))

    requested_methods = rec_cfg.get("methods", ["parametric_inverse", "latent_ridge"])
    methods = [str(m) for m in requested_methods if str(m) in RECON_METHODS]
    if not methods:
        raise RuntimeError(f"No valid reconstruction methods configured. Allowed: {sorted(RECON_METHODS)}")

    repeat_seeds = [int(v) for v in rec_cfg.get("repeat_seeds", cfg["ml"].get("repeat_seeds", [42]))]
    repeat_seeds = sorted(set(repeat_seeds))

    n_total = x.shape[0]
    n_strata = len(np.unique(strata))
    test_n = _compute_stratified_test_n(n_total, n_strata, float(cfg["ml"].get("test_size", 0.2)))

    repeat_rows: list[dict[str, float]] = []
    for method in methods:
        for seed in repeat_seeds:
            x_train, x_test, y_shape_train, y_shape_test, y_params_train, y_params_test, y_train, y_test = train_test_split(
                x,
                y_shape,
                y_params,
                y_flat,
                test_size=test_n,
                random_state=seed,
                stratify=strata,
            )

            fitted = fit_reconstruction_method(
                method=method,
                x_train=x_train,
                y_shape_train=y_shape_train,
                y_params_train=y_params_train,
                y_train_flat=y_train,
                cfg=cfg,
                seed=seed,
            )
            y_pred, aux = predict_reconstruction_images(
                method=method,
                fitted_model=fitted,
                x_test=x_test,
                cfg=cfg,
                image_height=h_img,
                image_width=w_img,
            )
            metrics = _evaluate_prediction(y_true=y_test, y_pred=y_pred, threshold=obstacle_threshold)

            shape_acc = 0.0
            dy_mae = 0.0
            eps_mae = 0.0
            if "shape_pred" in aux:
                shape_acc = float(np.mean(aux["shape_pred"] == y_shape_test))
            if "dy_pred" in aux:
                dy_mae = float(np.mean(np.abs(aux["dy_pred"] - y_params_test[:, 0])))
            if "eps_pred" in aux:
                eps_mae = float(np.mean(np.abs(aux["eps_pred"] - y_params_test[:, 1])))

            repeat_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "train_size": int(x_train.shape[0]),
                    "test_size": int(x_test.shape[0]),
                    "mse": float(metrics["mse"]),
                    "iou": float(metrics["iou"]),
                    "dice": float(metrics["dice"]),
                    "miou3": float(metrics["miou3"]),
                    "shape_acc": shape_acc,
                    "dy_mae": dy_mae,
                    "eps_mae": eps_mae,
                }
            )

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    repeat_df = pd.DataFrame(repeat_rows).sort_values(["method", "seed"])
    repeat_df.to_csv(reports_dir / "reconstruction_repeats.csv", index=False)

    method_summary_df = _summarize_by_method(repeat_df)
    method_summary_df.to_csv(reports_dir / "reconstruction_method_leaderboard.csv", index=False)
    _plot_method_comparison(method_summary_df, reports_dir / "reconstruction_method_comparison.png")

    best_method = str(
        method_summary_df.sort_values(["miou3_mean", "iou_mean", "dice_mean", "mse_mean"], ascending=[False, False, False, True]).iloc[0]["method"]
    )
    logger.info("Selected best reconstruction method: %s", best_method)

    base_seed = int(cfg["ml"].get("random_state", 42))
    x_train, x_test, y_shape_train, y_shape_test, y_params_train, y_params_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y_shape,
        y_params,
        y_flat,
        np.arange(n_total),
        test_size=test_n,
        random_state=base_seed,
        stratify=strata,
    )

    best_fit = fit_reconstruction_method(
        method=best_method,
        x_train=x_train,
        y_shape_train=y_shape_train,
        y_params_train=y_params_train,
        y_train_flat=y_train,
        cfg=cfg,
        seed=base_seed,
    )
    y_pred, aux = predict_reconstruction_images(
        method=best_method,
        fitted_model=best_fit,
        x_test=x_test,
        cfg=cfg,
        image_height=h_img,
        image_width=w_img,
    )
    base_metrics = _evaluate_prediction(y_true=y_test, y_pred=y_pred, threshold=obstacle_threshold)

    y_true_img = y_test.reshape(-1, h_img, w_img)
    y_pred_img = y_pred.reshape(-1, h_img, w_img)
    case_ids = features_df.iloc[idx_test]["case_id"].astype(str).tolist()
    pred_shape = aux.get("shape_pred")
    pred_shape_list = pred_shape.tolist() if pred_shape is not None else None

    _plot_examples(
        y_true_img=y_true_img,
        y_pred_img=y_pred_img,
        case_ids=case_ids,
        output_path=reports_dir / "reconstruction_examples.png",
        method_name=best_method,
        pred_shape=pred_shape_list,
    )
    _plot_iou_vs_eps(
        eps=features_df.iloc[idx_test]["eps"].to_numpy(dtype=float),
        iou=base_metrics["iou_values"],
        output_path=reports_dir / "reconstruction_iou_vs_eps.png",
    )

    case_metrics_df = _build_case_metrics_df(
        features_df=features_df,
        idx_test=np.asarray(idx_test),
        y_shape_test=np.asarray(y_shape_test),
        y_params_test=np.asarray(y_params_test),
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        metrics=base_metrics,
        aux=aux,
    )
    case_metrics_path = reports_dir / "reconstruction_case_metrics.csv"
    case_metrics_df.to_csv(case_metrics_path, index=False)

    random_metrics = _random_pair_metrics(y_true=y_test, y_pred=y_pred, threshold=obstacle_threshold, seed=123)
    _write_sanity_report(
        output_path=reports_dir / "reconstruction_sanity.md",
        aligned_metrics=base_metrics,
        random_metrics=random_metrics,
        case_metrics_df=case_metrics_df,
        confidence_threshold=float(rec_cfg.get("confidence_threshold", 0.45)),
    )
    shape_conf = aux.get("shape_confidence")
    if shape_conf is not None:
        _plot_confidence_vs_iou(
            shape_confidence=np.asarray(shape_conf, dtype=float),
            iou_values=np.asarray(base_metrics["iou_values"], dtype=float),
            output_path=reports_dir / "reconstruction_confidence_vs_iou.png",
        )

    re_rows = []
    for re_test in sorted(features_df["Re"].unique()):
        train_mask = features_df["Re"].to_numpy() != re_test
        test_mask = ~train_mask

        x_tr = x[train_mask]
        x_te = x[test_mask]
        y_shape_tr = y_shape[train_mask]
        y_shape_te = y_shape[test_mask]
        y_params_tr = y_params[train_mask]
        y_params_te = y_params[test_mask]
        y_tr = y_flat[train_mask]
        y_te = y_flat[test_mask]

        fit_re = fit_reconstruction_method(
            method=best_method,
            x_train=x_tr,
            y_shape_train=y_shape_tr,
            y_params_train=y_params_tr,
            y_train_flat=y_tr,
            cfg=cfg,
            seed=int(1000 + re_test),
        )
        y_hat, aux_re = predict_reconstruction_images(
            method=best_method,
            fitted_model=fit_re,
            x_test=x_te,
            cfg=cfg,
            image_height=h_img,
            image_width=w_img,
        )
        m_re = _evaluate_prediction(y_true=y_te, y_pred=y_hat, threshold=obstacle_threshold)

        shape_acc = 0.0
        dy_mae = 0.0
        eps_mae = 0.0
        if "shape_pred" in aux_re:
            shape_acc = float(np.mean(aux_re["shape_pred"] == y_shape_te))
        if "dy_pred" in aux_re:
            dy_mae = float(np.mean(np.abs(aux_re["dy_pred"] - y_params_te[:, 0])))
        if "eps_pred" in aux_re:
            eps_mae = float(np.mean(np.abs(aux_re["eps_pred"] - y_params_te[:, 1])))

        re_rows.append(
            {
                "Re_test": int(re_test),
                "n_test": int(y_te.shape[0]),
                "mse": float(m_re["mse"]),
                "iou": float(m_re["iou"]),
                "dice": float(m_re["dice"]),
                "miou3": float(m_re["miou3"]),
                "shape_acc": shape_acc,
                "dy_mae": dy_mae,
                "eps_mae": eps_mae,
            }
        )

    _write_summary(
        output_path=reports_dir / "reconstruction_summary.md",
        selected_method=best_method,
        method_summary_df=method_summary_df,
        test_n=test_n,
        train_n=n_total - test_n,
        repeat_seeds=repeat_seeds,
        re_rows=re_rows,
    )

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "reconstructor.pkl"

    model_blob = {
        "method": best_method,
        "feature_columns": feature_cols,
        "image_height": h_img,
        "image_width": w_img,
        "threshold": obstacle_threshold,
        "summary": method_summary_df.to_dict(orient="records"),
        "selected_metrics": {
            "mse": float(base_metrics["mse"]),
            "iou": float(base_metrics["iou"]),
            "dice": float(base_metrics["dice"]),
            "miou3": float(base_metrics["miou3"]),
            "random_pair_iou": float(random_metrics["iou"]),
            "random_pair_dice": float(random_metrics["dice"]),
            "random_pair_miou3": float(random_metrics["miou3"]),
            "shape_acc": float(np.mean(aux["shape_pred"] == y_shape_test)) if "shape_pred" in aux else None,
            "dy_mae": float(np.mean(np.abs(aux["dy_pred"] - y_params_test[:, 0]))) if "dy_pred" in aux else None,
            "eps_mae": float(np.mean(np.abs(aux["eps_pred"] - y_params_test[:, 1]))) if "eps_pred" in aux else None,
        },
        "config": cfg,
        "fitted_model": best_fit,
    }
    if best_method == "latent_ridge":
        # Backward compatibility for scripts expecting these keys.
        model_blob["scaler"] = best_fit["scaler"]
        model_blob["pca_y"] = best_fit["pca_y"]
        model_blob["regressor"] = best_fit["regressor"]
    if best_method == "parametric_inverse":
        model_blob["classifier"] = best_fit["classifier"]
        model_blob["param_regressor"] = best_fit["param_regressor"]

    with model_path.open("wb") as handle:
        pickle.dump(model_blob, handle)

    logger.info(
        "Reconstruction done. method=%s mse=%.6f iou=%.4f dice=%.4f miou3=%.4f model=%s",
        best_method,
        base_metrics["mse"],
        base_metrics["iou"],
        base_metrics["dice"],
        base_metrics["miou3"],
        model_path,
    )


if __name__ == "__main__":
    main()
