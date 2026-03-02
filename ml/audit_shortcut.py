from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.reconstruct import (
    _compute_stratified_test_n,
    _evaluate_prediction,
    _prepare_xy,
    _render_from_predicted_params,
    _clip_params,
)
from sim.config import load_config, repo_root
from sim.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit possible pattern-matching shortcuts in reconstruction")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--n_perm", type=int, default=30, help="Number of target-permutation trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for audit")
    return parser.parse_args()


def _fit_predict_parametric(
    x_train: np.ndarray,
    y_shape_train: np.ndarray,
    y_params_train: np.ndarray,
    x_test: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rec_cfg = cfg["reconstruction"]
    n_estimators = int(rec_cfg.get("audit_n_estimators", rec_cfg.get("param_n_estimators", 600)))
    min_samples_leaf = int(rec_cfg.get("param_min_samples_leaf", 1))
    image_height = int(rec_cfg["image_height"])
    image_width = int(rec_cfg["image_width"])

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

    shape_pred = classifier.predict(x_test)
    params_pred = regressor.predict(x_test)
    if params_pred.ndim == 1:
        params_pred = params_pred.reshape(-1, 2)
    dy_pred, eps_pred = _clip_params(params_pred[:, 0], params_pred[:, 1], cfg)

    y_pred = _render_from_predicted_params(
        shape_pred=shape_pred,
        dy_pred=dy_pred,
        eps_pred=eps_pred,
        cfg=cfg,
        image_height=image_height,
        image_width=image_width,
    ).reshape(x_test.shape[0], -1)

    shape_conf = None
    if hasattr(classifier, "predict_proba"):
        shape_conf = np.max(classifier.predict_proba(x_test), axis=1)
    return y_pred, {
        "shape_pred": shape_pred,
        "dy_pred": dy_pred,
        "eps_pred": eps_pred,
        "shape_confidence": shape_conf,
    }


def _random_pair_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, seed: int = 123) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_true.shape[0])
    return _evaluate_prediction(y_true=y_true[perm], y_pred=y_pred, threshold=threshold)


def _nearest_neighbor_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    threshold: float,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(x_train_s)
    dists, idx = nn.kneighbors(x_test_s, return_distance=True)
    idx = idx.reshape(-1)
    dists = dists.reshape(-1)

    y_pred = y_train[idx]
    metrics = _evaluate_prediction(y_true=y_test, y_pred=y_pred, threshold=threshold)
    metrics["dup_like_rate"] = float(np.mean(dists < 1e-9))
    metrics["nn_dist_mean"] = float(np.mean(dists))
    metrics["nn_dist_p05"] = float(np.quantile(dists, 0.05))
    return metrics, dists


def _plot_null_distribution(null_df: pd.DataFrame, observed_iou: float, nn_iou: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(null_df["iou"], bins=16, alpha=0.7, color="#93c5fd", edgecolor="#1e3a8a")
    ax.axvline(observed_iou, color="#16a34a", lw=2.2, label=f"observed={observed_iou:.3f}")
    ax.axvline(nn_iou, color="#b45309", lw=2.0, ls="--", label=f"1-NN={nn_iou:.3f}")
    ax.set_xlabel("Obstacle IoU under target permutation")
    ax.set_ylabel("Count")
    ax.set_title("Permutation Null Distribution (Pattern-Matching Audit)")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    cfg_name: str,
    observed: dict[str, float],
    random_pair: dict[str, float],
    nn_metrics: dict[str, float],
    null_df: pd.DataFrame,
    p_iou: float,
    p_miou3: float,
) -> None:
    lines: list[str] = []
    lines.append("# Pattern-Matching Audit")
    lines.append("")
    lines.append(f"- Config: `{cfg_name}`")
    lines.append(f"- Permutations: `{len(null_df)}`")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append(
        f"- Observed reconstruction: IoU={observed['iou']:.4f}, Dice={observed['dice']:.4f}, mIoU3={observed['miou3']:.4f}"
    )
    lines.append(
        f"- Random-pair control: IoU={random_pair['iou']:.4f}, Dice={random_pair['dice']:.4f}, mIoU3={random_pair['miou3']:.4f}"
    )
    lines.append(
        f"- 1-NN template baseline: IoU={nn_metrics['iou']:.4f}, Dice={nn_metrics['dice']:.4f}, mIoU3={nn_metrics['miou3']:.4f}"
    )
    lines.append("")
    lines.append("## Permutation Test (H0: input-target mapping is random)")
    lines.append(
        f"- Null IoU mean±std: {null_df['iou'].mean():.4f} ± {null_df['iou'].std(ddof=0):.4f}; p-value={p_iou:.6f}"
    )
    lines.append(
        f"- Null mIoU3 mean±std: {null_df['miou3'].mean():.4f} ± {null_df['miou3'].std(ddof=0):.4f}; p-value={p_miou3:.6f}"
    )
    lines.append("")
    lines.append("## Duplicate-Like Risk")
    lines.append(f"- Near-duplicate rate (scaled NN dist < 1e-9): {nn_metrics['dup_like_rate']:.6f}")
    lines.append(f"- NN distance mean: {nn_metrics['nn_dist_mean']:.6f}")
    lines.append(f"- NN distance p05: {nn_metrics['nn_dist_p05']:.6f}")
    lines.append("")
    lines.append("## Conclusion")
    if p_iou < 0.01 and observed["iou"] > nn_metrics["iou"] and observed["iou"] > random_pair["iou"]:
        lines.append("- Current evidence rejects pure template pattern-matching as the primary explanation.")
    else:
        lines.append("- Shortcut risk remains non-negligible; strengthen controls or expand OOD tests.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("audit", root / "logs" / "audit.log")

    features_path = root / "data" / "features" / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run dataset and feature extraction first.")

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        raise RuntimeError("features.csv is empty")

    x, y_shape, y_params, _y_img, y_flat, strata, _feature_cols = _prepare_xy(features_df, cfg)
    test_n = _compute_stratified_test_n(
        n_total=x.shape[0],
        n_strata=len(np.unique(strata)),
        requested_ratio=float(cfg["ml"].get("test_size", 0.2)),
    )

    x_train, x_test, y_shape_train, y_shape_test, y_params_train, y_params_test, y_train, y_test = train_test_split(
        x,
        y_shape,
        y_params,
        y_flat,
        test_size=test_n,
        random_state=int(args.seed),
        stratify=strata,
    )

    threshold = float(cfg["reconstruction"].get("obstacle_threshold", 0.8))

    y_pred, aux = _fit_predict_parametric(
        x_train=x_train,
        y_shape_train=y_shape_train,
        y_params_train=y_params_train,
        x_test=x_test,
        cfg=cfg,
        seed=int(args.seed),
    )
    observed = _evaluate_prediction(y_true=y_test, y_pred=y_pred, threshold=threshold)
    random_pair = _random_pair_metrics(y_true=y_test, y_pred=y_pred, threshold=threshold, seed=int(args.seed) + 1)
    nn_metrics, nn_dists = _nearest_neighbor_baseline(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        threshold=threshold,
        y_test=y_test,
    )

    rng = np.random.default_rng(int(args.seed) + 2)
    null_rows: list[dict[str, float]] = []
    for i in range(int(args.n_perm)):
        perm_idx = rng.permutation(y_train.shape[0])
        y_shape_perm = y_shape_train[perm_idx]
        y_params_perm = y_params_train[perm_idx]

        y_perm_pred, _ = _fit_predict_parametric(
            x_train=x_train,
            y_shape_train=y_shape_perm,
            y_params_train=y_params_perm,
            x_test=x_test,
            cfg=cfg,
            seed=int(args.seed) + 100 + i,
        )
        m = _evaluate_prediction(y_true=y_test, y_pred=y_perm_pred, threshold=threshold)
        null_rows.append(
            {
                "perm_idx": int(i),
                "iou": float(m["iou"]),
                "dice": float(m["dice"]),
                "miou3": float(m["miou3"]),
            }
        )
        logger.info("Permutation %d/%d done. iou=%.4f", i + 1, int(args.n_perm), float(m["iou"]))

    null_df = pd.DataFrame(null_rows)
    p_iou = float((1.0 + np.sum(null_df["iou"].to_numpy() >= observed["iou"])) / (len(null_df) + 1.0))
    p_miou3 = float((1.0 + np.sum(null_df["miou3"].to_numpy() >= observed["miou3"])) / (len(null_df) + 1.0))

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    null_csv = reports_dir / "pattern_audit_null.csv"
    null_df.to_csv(null_csv, index=False)

    nn_df = pd.DataFrame({"nn_distance": nn_dists})
    nn_csv = reports_dir / "pattern_audit_nn_distance.csv"
    nn_df.to_csv(nn_csv, index=False)

    _plot_null_distribution(
        null_df=null_df,
        observed_iou=float(observed["iou"]),
        nn_iou=float(nn_metrics["iou"]),
        output_path=reports_dir / "pattern_audit_null_hist.png",
    )

    report_md = reports_dir / "pattern_audit.md"
    _write_report(
        path=report_md,
        cfg_name=str(Path(args.config).name),
        observed=observed,
        random_pair=random_pair,
        nn_metrics=nn_metrics,
        null_df=null_df,
        p_iou=p_iou,
        p_miou3=p_miou3,
    )

    logger.info(
        "Pattern audit done. obs_iou=%.4f null_iou_mean=%.4f p=%.6f report=%s",
        float(observed["iou"]),
        float(null_df["iou"].mean()),
        p_iou,
        report_md,
    )


if __name__ == "__main__":
    main()
