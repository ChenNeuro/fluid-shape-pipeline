from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sim.config import load_config, repo_root
from sim.geometry_mask import render_case_image
from sim.logging_utils import setup_logger


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct geometry images from probe features")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def _prepare_xy(features_df: pd.DataFrame, cfg: dict):
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    x = features_df[feature_cols].to_numpy(dtype=float)

    sim_cfg = cfg["simulation"]
    rec_cfg = cfg["reconstruction"]

    images = []
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
        images.append(img)

    y_img = np.stack(images, axis=0)
    y = y_img.reshape(y_img.shape[0], -1)

    strata = (features_df["shape"].astype(str) + "_Re" + features_df["Re"].astype(str)).to_numpy()
    return x, y, y_img, strata, feature_cols


def _compute_stratified_test_n(n_total: int, n_strata: int, requested_ratio: float) -> int:
    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    return min(max(requested_test_n, min_test_n), max_test_n)


def _iou_and_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float]:
    yt = y_true >= threshold
    yp = y_pred >= threshold
    inter = np.logical_and(yt, yp).sum()
    union = np.logical_or(yt, yp).sum()
    dice_den = yt.sum() + yp.sum()

    iou = float(inter / (union + 1e-9))
    dice = float((2.0 * inter) / (dice_den + 1e-9))
    return iou, dice


def _fit_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    latent_dim: int,
    ridge_alpha: float,
):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    n_components = max(2, min(latent_dim, y_train.shape[0] - 1, y_train.shape[1]))
    pca_y = PCA(n_components=n_components, random_state=0)
    z_train = pca_y.fit_transform(y_train)

    reg = Ridge(alpha=ridge_alpha)
    reg.fit(x_train_s, z_train)

    z_pred = reg.predict(x_test_s)
    y_pred = pca_y.inverse_transform(z_pred)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return scaler, pca_y, reg, y_pred


def _plot_examples(
    y_true_img: np.ndarray,
    y_pred_img: np.ndarray,
    case_ids: list[str],
    output_path: Path,
    n_show: int = 6,
) -> None:
    n = min(n_show, y_true_img.shape[0])
    idx = np.linspace(0, y_true_img.shape[0] - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(10, 2.3 * n))
    if n == 1:
        axes = np.array([axes])

    for row, i in enumerate(idx):
        gt = y_true_img[i]
        pdm = y_pred_img[i]
        err = np.abs(gt - pdm)

        axes[row, 0].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"GT {case_ids[i]}")
        axes[row, 1].imshow(pdm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("Pred")
        axes[row, 2].imshow(err, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("|Err|")

        for col in range(3):
            axes[row, col].axis("off")

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


def _summarize_repeats(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_repeats": int(df.shape[0]),
        "mse_mean": float(df["mse"].mean()),
        "mse_std": float(df["mse"].std(ddof=0)),
        "iou_mean": float(df["iou"].mean()),
        "iou_std": float(df["iou"].std(ddof=0)),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std(ddof=0)),
    }


def _write_summary(
    output_path: Path,
    summary: dict[str, float],
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
    lines.append("")
    lines.append("## Repeated Holdout Metrics")
    lines.append(f"- MSE mean±std: {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")
    lines.append(f"- IoU mean±std: {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
    lines.append(f"- Dice mean±std: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    lines.append("")
    lines.append("## Leave-One-Re-Out")
    for row in re_rows:
        lines.append(
            f"- Re={row['Re_test']} (n={row['n_test']}): MSE={row['mse']:.6f}, IoU={row['iou']:.4f}, Dice={row['dice']:.4f}"
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

    x, y, y_img, strata, feature_cols = _prepare_xy(features_df, cfg)

    rec_cfg = cfg["reconstruction"]
    h_img = int(rec_cfg["image_height"])
    w_img = int(rec_cfg["image_width"])
    latent_dim = int(rec_cfg.get("latent_dim", 64))
    ridge_alpha = float(rec_cfg.get("ridge_alpha", 1.0))
    obstacle_threshold = float(rec_cfg.get("obstacle_threshold", 0.8))

    repeat_seeds = [int(v) for v in rec_cfg.get("repeat_seeds", cfg["ml"].get("repeat_seeds", [42]))]
    repeat_seeds = sorted(set(repeat_seeds))

    n_total = x.shape[0]
    n_strata = len(np.unique(strata))
    test_n = _compute_stratified_test_n(n_total, n_strata, float(cfg["ml"].get("test_size", 0.2)))

    repeat_rows = []
    for seed in repeat_seeds:
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x,
            y,
            np.arange(n_total),
            test_size=test_n,
            random_state=seed,
            stratify=strata,
        )
        _, _, _, y_pred = _fit_and_predict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            latent_dim=latent_dim,
            ridge_alpha=ridge_alpha,
        )

        mse = float(mean_squared_error(y_test, y_pred))
        iou_list = []
        dice_list = []
        for i in range(y_test.shape[0]):
            iou, dice = _iou_and_dice(y_test[i], y_pred[i], threshold=obstacle_threshold)
            iou_list.append(iou)
            dice_list.append(dice)

        repeat_rows.append(
            {
                "seed": int(seed),
                "train_size": int(x_train.shape[0]),
                "test_size": int(x_test.shape[0]),
                "mse": mse,
                "iou": float(np.mean(iou_list)),
                "dice": float(np.mean(dice_list)),
            }
        )

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    repeat_df = pd.DataFrame(repeat_rows).sort_values("seed")
    repeat_df.to_csv(reports_dir / "reconstruction_repeats.csv", index=False)
    repeat_summary = _summarize_repeats(repeat_df)

    base_seed = int(cfg["ml"].get("random_state", 42))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        np.arange(n_total),
        test_size=test_n,
        random_state=base_seed,
        stratify=strata,
    )
    scaler, pca_y, reg, y_pred = _fit_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        latent_dim=latent_dim,
        ridge_alpha=ridge_alpha,
    )

    y_true_img = y_test.reshape(-1, h_img, w_img)
    y_pred_img = y_pred.reshape(-1, h_img, w_img)
    case_ids = features_df.iloc[idx_test]["case_id"].astype(str).tolist()

    _plot_examples(
        y_true_img=y_true_img,
        y_pred_img=y_pred_img,
        case_ids=case_ids,
        output_path=reports_dir / "reconstruction_examples.png",
    )

    per_case_iou = []
    per_case_dice = []
    for i in range(y_test.shape[0]):
        iou, dice = _iou_and_dice(y_test[i], y_pred[i], threshold=obstacle_threshold)
        per_case_iou.append(iou)
        per_case_dice.append(dice)

    eps_test = features_df.iloc[idx_test]["eps"].to_numpy(dtype=float)
    _plot_iou_vs_eps(
        eps=eps_test,
        iou=np.array(per_case_iou, dtype=float),
        output_path=reports_dir / "reconstruction_iou_vs_eps.png",
    )

    re_rows = []
    for re_test in sorted(features_df["Re"].unique()):
        train_df = features_df[features_df["Re"] != re_test]
        test_df = features_df[features_df["Re"] == re_test]

        x_tr = train_df[feature_cols].to_numpy(dtype=float)
        x_te = test_df[feature_cols].to_numpy(dtype=float)

        y_tr_img = []
        y_te_img = []
        for _, row in train_df.iterrows():
            y_tr_img.append(
                render_case_image(
                    shape=str(row["shape"]),
                    dy=float(row["dy"]),
                    eps=float(row["eps"]),
                    h=float(cfg["simulation"]["H"]),
                    d_ratio=float(cfg["simulation"]["d_ratio"]),
                    x0=float(cfg["simulation"]["x0"]),
                    y0=float(cfg["simulation"]["y0"]),
                    l_in=float(cfg["simulation"]["L_in"]),
                    l_out=float(cfg["simulation"]["L_out"]),
                    image_height=h_img,
                    image_width=w_img,
                    eps_max_for_canvas=float(cfg["simulation"]["perturb"]["eps_max"]),
                )
            )
        for _, row in test_df.iterrows():
            y_te_img.append(
                render_case_image(
                    shape=str(row["shape"]),
                    dy=float(row["dy"]),
                    eps=float(row["eps"]),
                    h=float(cfg["simulation"]["H"]),
                    d_ratio=float(cfg["simulation"]["d_ratio"]),
                    x0=float(cfg["simulation"]["x0"]),
                    y0=float(cfg["simulation"]["y0"]),
                    l_in=float(cfg["simulation"]["L_in"]),
                    l_out=float(cfg["simulation"]["L_out"]),
                    image_height=h_img,
                    image_width=w_img,
                    eps_max_for_canvas=float(cfg["simulation"]["perturb"]["eps_max"]),
                )
            )

        y_tr = np.stack(y_tr_img, axis=0).reshape(len(y_tr_img), -1)
        y_te = np.stack(y_te_img, axis=0).reshape(len(y_te_img), -1)

        _, _, _, y_hat = _fit_and_predict(
            x_train=x_tr,
            y_train=y_tr,
            x_test=x_te,
            latent_dim=latent_dim,
            ridge_alpha=ridge_alpha,
        )

        mse = float(mean_squared_error(y_te, y_hat))
        ious = []
        dices = []
        for i in range(y_te.shape[0]):
            iou, dice = _iou_and_dice(y_te[i], y_hat[i], threshold=obstacle_threshold)
            ious.append(iou)
            dices.append(dice)

        re_rows.append(
            {
                "Re_test": int(re_test),
                "n_test": int(y_te.shape[0]),
                "mse": mse,
                "iou": float(np.mean(ious)),
                "dice": float(np.mean(dices)),
            }
        )

    _write_summary(
        output_path=reports_dir / "reconstruction_summary.md",
        summary=repeat_summary,
        test_n=test_n,
        train_n=n_total - test_n,
        repeat_seeds=repeat_seeds,
        re_rows=re_rows,
    )

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "reconstructor.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "scaler": scaler,
                "pca_y": pca_y,
                "regressor": reg,
                "feature_columns": feature_cols,
                "image_height": h_img,
                "image_width": w_img,
                "threshold": obstacle_threshold,
                "summary": repeat_summary,
                "config": cfg,
            },
            handle,
        )

    logger.info(
        "Reconstruction done. mse=%.6f iou=%.4f dice=%.4f model=%s",
        repeat_summary["mse_mean"],
        repeat_summary["iou_mean"],
        repeat_summary["dice_mean"],
        model_path,
    )


if __name__ == "__main__":
    main()
