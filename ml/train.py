from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from sim.config import load_config, repo_root
from sim.data_schema import find_probes_csv
from sim.logging_utils import setup_logger


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline classifier and generate reports")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def _prepare_xy(features_df: pd.DataFrame):
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    x = features_df[feature_cols].to_numpy(dtype=float)
    y = features_df["shape"].to_numpy()
    return x, y, feature_cols


def _plot_pca(features_df: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    x = features_df[feature_cols].to_numpy(dtype=float)
    y = features_df["shape"].to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=0)
    x_pca = pca.fit_transform(x_scaled)

    plt.figure(figsize=(8, 6))
    for shape in sorted(np.unique(y)):
        mask = y == shape
        plt.scatter(x_pca[mask, 0], x_pca[mask, 1], label=shape, alpha=0.8)
    plt.title("PCA Separability of Case Features")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _select_typical_cases(features_df: pd.DataFrame) -> dict[str, str]:
    target_re = float(np.median(features_df["Re"]))
    selected = {}

    for shape in sorted(features_df["shape"].unique()):
        subset = features_df[features_df["shape"] == shape].copy()
        subset["score"] = (
            (subset["Re"].astype(float) - target_re).abs() / max(target_re, 1.0)
            + subset["dy"].astype(float).abs() * 10.0
            + subset["eps"].astype(float).abs() * 10.0
        )
        chosen = subset.sort_values("score").iloc[0]
        selected[shape] = str(chosen["case_id"])

    return selected


def _plot_spectra_examples(features_df: pd.DataFrame, raw_root: Path, output_path: Path) -> None:
    cases = _select_typical_cases(features_df)
    n_shapes = len(cases)
    fig, axes = plt.subplots(1, n_shapes, figsize=(5 * n_shapes, 4), squeeze=False)

    for idx, (shape, case_id) in enumerate(cases.items()):
        ax = axes[0, idx]
        raw_csv = find_probes_csv(raw_root / case_id)
        df = pd.read_csv(raw_csv)
        probe_cols = [c for c in df.columns if c.startswith("u_")]
        probe_col = probe_cols[len(probe_cols) // 2]

        t = df["time"].to_numpy(dtype=float)
        x = df[probe_col].to_numpy(dtype=float)
        x = x - np.mean(x)
        dt = float(np.median(np.diff(t)))

        fft_vals = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=dt)
        amps = np.abs(fft_vals) * (2.0 / x.size)

        mask = freqs <= min(8.0, np.max(freqs))
        ax.plot(freqs[mask], amps[mask], lw=1.6)
        ax.set_title(f"{shape}: {case_id}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)

    fig.suptitle("Representative Probe Spectra by Shape")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Baseline Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _robustness_curve(x: np.ndarray, y: np.ndarray, eps: np.ndarray, model_cfg: dict, output_path: Path) -> list[dict[str, float]]:
    folds = int(model_cfg.get("cv_folds", 5))
    seed = int(model_cfg.get("random_state", 42))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    y_pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)

    abs_eps = np.abs(eps.astype(float))
    eps_max = float(np.max(abs_eps)) if abs_eps.size else 0.0
    if eps_max == 0.0:
        bins = np.array([0.0, 1.0])
    else:
        n_bins = int(model_cfg.get("robustness_bins", 6))
        bins = np.linspace(0.0, eps_max, n_bins + 1)

    centers = []
    accuracies = []
    counts = []
    rows = []

    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == len(bins) - 2:
            mask = (abs_eps >= lo) & (abs_eps <= hi)
        else:
            mask = (abs_eps >= lo) & (abs_eps < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue

        acc = float(accuracy_score(y[mask], y_pred[mask]))
        center = float((lo + hi) / 2.0)

        centers.append(center)
        accuracies.append(acc)
        counts.append(count)
        rows.append({"eps_bin_center": center, "accuracy": acc, "count": count})

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(centers, accuracies, marker="o", lw=2)
    ax1.set_xlabel("|eps| (outlet lens perturbation amplitude)")
    ax1.set_ylabel("Cross-validated accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=(centers[1] - centers[0]) * 0.6 if len(centers) > 1 else 0.001, alpha=0.2)
    ax2.set_ylabel("Samples per bin")

    plt.title("Robustness Sweep: Geometry Perturbation vs Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    return rows


def _leave_one_re_out(features_df: pd.DataFrame, feature_cols: list[str], model_cfg: dict) -> list[dict[str, float]]:
    results = []
    seed = int(model_cfg.get("random_state", 42))

    for re_value in sorted(features_df["Re"].unique()):
        train_df = features_df[features_df["Re"] != re_value]
        test_df = features_df[features_df["Re"] == re_value]

        if train_df.empty or test_df.empty:
            continue

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["shape"].to_numpy()
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["shape"].to_numpy()

        model = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        results.append(
            {
                "Re_test": int(re_value),
                "n_test": int(test_df.shape[0]),
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )

    return results


def _write_summary(
    output_md: Path,
    cfg: dict,
    features_df: pd.DataFrame,
    split_metrics: dict,
    split_meta: dict,
    re_results: list[dict[str, float]],
    robust_rows: list[dict[str, float]],
    top_importance: list[tuple[str, float]],
    importance_method: str,
    robustness_note: str,
    feature_note: str,
    manifest_df: pd.DataFrame,
) -> None:
    sim_cfg = cfg["simulation"]
    success_cases = int((manifest_df["status"] == "success").sum())
    failed_cases = int((manifest_df["status"] == "failed").sum())
    total_cases = int(manifest_df.shape[0])
    failure_rate = (failed_cases / total_cases * 100.0) if total_cases else 0.0

    lines = []
    lines.append("# Obstacle Shape Identification Summary")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(f"- Solver mode: `{cfg['project'].get('solver', 'synthetic')}` (this run used synthetic baseline)")
    lines.append(
        f"- Shapes: {', '.join(sim_cfg['shapes'])}; Re set: {sim_cfg['re_values']}; perturbations per (shape, Re): {sim_cfg['perturbations_per_combo']}"
    )
    lines.append(
        f"- Probes: N={sim_cfg['probes_n']} at outlet x=L; geometry perturbation: dy in [{sim_cfg['perturb']['dy_min']}, {sim_cfg['perturb']['dy_max']}] * H, eps in [0, {sim_cfg['perturb']['eps_max']}]"
    )
    lines.append(f"- Data rows in `features.csv`: {features_df.shape[0]}, feature dimensions: {features_df.shape[1] - len(META_COLS)}")
    lines.append(f"- Case execution: total={total_cases}, success={success_cases}, failed={failed_cases}, failure rate={failure_rate:.2f}%")
    lines.append("")

    lines.append("## Baseline Split Metrics (Stratified Holdout)")
    lines.append(
        f"- Split strategy: stratified by `(shape, Re)` with train={split_meta['train_size']} test={split_meta['test_size']} (requested test ratio={split_meta['requested_test_ratio']:.3f}, applied ratio={split_meta['applied_test_ratio']:.3f})"
    )
    lines.append(f"- Accuracy: {split_metrics['accuracy']:.4f}")
    lines.append(f"- Macro F1: {split_metrics['macro_f1']:.4f}")
    lines.append("")

    lines.append("## Leave-One-Re-Out Generalization")
    for row in re_results:
        lines.append(
            f"- Train on Re != {row['Re_test']}, test on Re={row['Re_test']} (n={row['n_test']}): accuracy={row['accuracy']:.4f}, macro F1={row['macro_f1']:.4f}"
        )
    lines.append("")

    lines.append("## Robustness Sweep (|eps| bins)")
    for row in robust_rows:
        lines.append(
            f"- |eps|~{row['eps_bin_center']:.5f}: accuracy={row['accuracy']:.4f} (n={row['count']})"
        )
    lines.append("")

    lines.append("## Most Influential Features (Permutation Importance)")
    lines.append(f"- Importance method: {importance_method}")
    for name, score in top_importance:
        lines.append(f"- {name}: {score:.6f}")
    lines.append("")

    lines.append("## Failed Cases")
    failed_rows = manifest_df[manifest_df["status"] == "failed"]
    if failed_rows.empty:
        lines.append("- None")
    else:
        for _, row in failed_rows.iterrows():
            lines.append(f"- {row['case_id']}: {row['error']}")
    lines.append("")

    lines.append("## Key Conclusions")
    lines.append("- Outlet probe velocity signatures are separable across obstacle shapes with the current feature set.")
    lines.append(f"- {feature_note}")
    lines.append(f"- {robustness_note}")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("- Replace synthetic generator with OpenFOAM runs for physics-grounded validation while keeping raw CSV schema unchanged.")
    lines.append("- Add more challenging perturbations (x-shift, obstacle rotation, inlet profile changes) and domain randomization.")
    lines.append("- Benchmark temporal models (1D CNN / transformer) directly on probe sequences against feature-based baseline.")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("train", root / "logs" / "train.log")

    features_csv = root / "data" / "features" / "features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Missing features file: {features_csv}. Run make dataset first.")

    features_df = pd.read_csv(features_csv)
    if features_df.empty:
        raise RuntimeError("features.csv is empty")

    x, y, feature_cols = _prepare_xy(features_df)

    ml_cfg = cfg["ml"]
    random_state = int(ml_cfg.get("random_state", 42))

    strata = features_df["shape"].astype(str) + "_Re" + features_df["Re"].astype(str)
    n_total = int(features_df.shape[0])
    n_strata = int(strata.nunique())
    requested_ratio = float(ml_cfg.get("test_size", 0.2))

    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    test_n = min(max(requested_test_n, min_test_n), max_test_n)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_n,
        random_state=random_state,
        stratify=strata,
    )
    split_meta = {
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "requested_test_ratio": requested_ratio,
        "applied_test_ratio": float(x_test.shape[0] / n_total),
    }

    model = RandomForestClassifier(
        n_estimators=int(ml_cfg.get("n_estimators", 300)),
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(np.unique(y))
    _plot_confusion(y_test, y_pred, labels, reports_dir / "confusion_matrix.png")
    _plot_pca(features_df, feature_cols, reports_dir / "separability_pca.png")
    _plot_spectra_examples(features_df, root / "data" / "raw", reports_dir / "spectra_examples.png")
    robust_rows = _robustness_curve(
        x,
        y,
        features_df["eps"].to_numpy(dtype=float),
        ml_cfg,
        reports_dir / "robustness_sweep.png",
    )

    re_results = _leave_one_re_out(features_df, feature_cols, ml_cfg)

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=int(ml_cfg.get("perm_repeats", 20)),
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )
    importance_scores = perm.importances_mean
    importance_method = "permutation importance (test split)"
    if np.all(np.isclose(importance_scores, 0.0)):
        importance_scores = model.feature_importances_
        importance_method = "random forest impurity importance (fallback, permutation all-zero)"

    top_idx = np.argsort(importance_scores)[::-1][:10]
    top_importance = [(feature_cols[i], float(importance_scores[i])) for i in top_idx]

    top_feature_names = [name for name, _ in top_importance]
    n_freq = sum(("f_peak" in name) or ("band_" in name) for name in top_feature_names)
    if n_freq >= max(3, len(top_feature_names) // 2):
        feature_note = "Frequency-domain descriptors are the dominant contributors among top-ranked features."
    else:
        feature_note = "Top contributors are mixed across frequency, statistics, and cross-probe structure features."

    if len(robust_rows) >= 2:
        first_acc = robust_rows[0]["accuracy"]
        last_acc = robust_rows[-1]["accuracy"]
        delta = last_acc - first_acc
        if delta <= -0.05:
            robustness_note = "Accuracy drops at larger outlet lens perturbations, indicating sensitivity to geometry mismatch."
        elif delta >= 0.05:
            robustness_note = "Accuracy increases slightly in higher perturbation bins for this synthetic dataset."
        else:
            robustness_note = "Accuracy remains broadly stable across the tested outlet lens perturbation range."
    else:
        robustness_note = "Robustness trend is inconclusive because available perturbation bins are limited."

    manifest_df = pd.read_csv(root / "data" / "raw" / "manifest.csv")
    _write_summary(
        reports_dir / "summary.md",
        cfg,
        features_df,
        metrics,
        split_meta,
        re_results,
        robust_rows,
        top_importance,
        importance_method,
        robustness_note,
        feature_note,
        manifest_df,
    )

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_cols,
                "labels": labels,
                "metrics": metrics,
                "config": cfg,
            },
            handle,
        )

    logger.info(
        "Training done. accuracy=%.4f macro_f1=%.4f model=%s reports=%s",
        metrics["accuracy"],
        metrics["macro_f1"],
        model_path,
        reports_dir,
    )


if __name__ == "__main__":
    main()
