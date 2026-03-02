from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sim.config import load_config, repo_root
from sim.logging_utils import setup_logger


META_COLS = ["case_id", "shape", "Re", "dy", "eps", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOTA-candidate models and select the best")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def _prepare_xy(features_df: pd.DataFrame):
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    x = features_df[feature_cols].to_numpy(dtype=float)
    y = features_df["shape"].to_numpy()
    strata = (features_df["shape"].astype(str) + "_Re" + features_df["Re"].astype(str)).to_numpy()
    return x, y, strata, feature_cols


def _compute_stratified_test_n(n_total: int, n_strata: int, requested_ratio: float) -> int:
    requested_test_n = max(1, int(round(requested_ratio * n_total)))
    min_test_n = n_strata
    max_test_n = n_total - n_strata
    if max_test_n < 1:
        raise RuntimeError("Insufficient samples for stratification by (shape, Re)")
    return min(max(requested_test_n, min_test_n), max_test_n)


def _build_candidate_models(model_cfg: dict, seed: int) -> dict[str, object]:
    n_estimators = int(model_cfg.get("n_estimators", 400))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    et = ExtraTreesClassifier(
        n_estimators=max(600, int(1.5 * n_estimators)),
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(C=float(model_cfg.get("svc_c", 10.0)), kernel="rbf", gamma="scale")),
        ]
    )

    hard_vote = VotingClassifier(
        estimators=[("rf", clone(rf)), ("et", clone(et)), ("svc", clone(svc))],
        voting="hard",
        n_jobs=-1,
    )

    return {
        "rf": rf,
        "extratrees": et,
        "svc_rbf": svc,
        "hard_vote_ensemble": hard_vote,
    }


def _evaluate_repeats(
    model_name: str,
    model_builder,
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    test_n: int,
    seeds: list[int],
) -> tuple[list[dict], dict[str, float]]:
    rows = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_n,
            random_state=seed,
            stratify=strata,
        )

        model = model_builder(seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        rows.append(
            {
                "model": model_name,
                "seed": int(seed),
                "train_size": int(x_train.shape[0]),
                "test_size": int(x_test.shape[0]),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            }
        )

    acc = np.array([r["accuracy"] for r in rows], dtype=float)
    f1 = np.array([r["macro_f1"] for r in rows], dtype=float)
    summary = {
        "model": model_name,
        "n_repeats": len(rows),
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc, ddof=0)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1, ddof=0)),
    }
    return rows, summary


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _robustness_curve(model, x: np.ndarray, y: np.ndarray, eps: np.ndarray, model_cfg: dict, output_path: Path) -> list[dict[str, float]]:
    folds = int(model_cfg.get("cv_folds", 5))
    seed = int(model_cfg.get("random_state", 42))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    y_pred = cross_val_predict(model, x, y, cv=cv)

    abs_eps = np.abs(eps.astype(float))
    eps_max = float(np.max(abs_eps)) if abs_eps.size else 0.0
    if eps_max == 0.0:
        bins = np.array([0.0, 1.0])
    else:
        n_bins = int(model_cfg.get("robustness_bins", 6))
        bins = np.linspace(0.0, eps_max, n_bins + 1)

    rows = []
    centers = []
    accuracies = []
    counts = []

    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == len(bins) - 2:
            mask = (abs_eps >= lo) & (abs_eps <= hi)
        else:
            mask = (abs_eps >= lo) & (abs_eps < hi)

        n = int(np.sum(mask))
        if n == 0:
            continue

        acc = float(accuracy_score(y[mask], y_pred[mask]))
        center = float((lo + hi) / 2.0)
        rows.append({"eps_bin_center": center, "accuracy": acc, "count": n})
        centers.append(center)
        accuracies.append(acc)
        counts.append(n)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(centers, accuracies, marker="o", lw=2)
    ax1.set_xlabel("|eps| (outlet lens perturbation amplitude)")
    ax1.set_ylabel("Cross-validated accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    width = (centers[1] - centers[0]) * 0.6 if len(centers) > 1 else 0.001
    ax2.bar(centers, counts, width=width, alpha=0.2)
    ax2.set_ylabel("Samples per bin")

    plt.title("SOTA Candidate Robustness Sweep")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    return rows


def _leave_one_re_out(features_df: pd.DataFrame, feature_cols: list[str], model_builder) -> list[dict[str, float]]:
    rows = []
    for re_test in sorted(features_df["Re"].unique()):
        train_df = features_df[features_df["Re"] != re_test]
        test_df = features_df[features_df["Re"] == re_test]

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["shape"].to_numpy()
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["shape"].to_numpy()

        model = model_builder(int(1000 + re_test))
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        rows.append(
            {
                "Re_test": int(re_test),
                "n_test": int(test_df.shape[0]),
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )
    return rows


def _write_sota_summary(
    path: Path,
    best_model_name: str,
    best_seed: int,
    holdout_metrics: dict[str, float],
    leaderboard: pd.DataFrame,
    robust_rows: list[dict[str, float]],
    re_rows: list[dict[str, float]],
) -> None:
    lines = []
    lines.append("# SOTA Candidate Summary")
    lines.append("")
    lines.append(f"- Selected model: `{best_model_name}`")
    lines.append(f"- Selection seed (holdout fit): `{best_seed}`")
    lines.append(f"- Holdout accuracy: {holdout_metrics['accuracy']:.4f}")
    lines.append(f"- Holdout macro F1: {holdout_metrics['macro_f1']:.4f}")
    lines.append("")

    lines.append("## Candidate Leaderboard (Repeated Holdout)")
    for _, row in leaderboard.iterrows():
        lines.append(
            f"- {row['model']}: acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, macroF1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
        )
    lines.append("")

    lines.append("## Leave-One-Re-Out")
    for row in re_rows:
        lines.append(
            f"- Re={row['Re_test']}: acc={row['accuracy']:.4f}, macroF1={row['macro_f1']:.4f} (n={row['n_test']})"
        )
    lines.append("")

    lines.append("## Robustness Sweep")
    for row in robust_rows:
        lines.append(f"- |eps|~{row['eps_bin_center']:.5f}: acc={row['accuracy']:.4f} (n={row['count']})")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("train_sota", root / "logs" / "train_sota.log")

    features_csv = root / "data" / "features" / "features.csv"
    features_df = pd.read_csv(features_csv)

    x, y, strata, feature_cols = _prepare_xy(features_df)

    ml_cfg = cfg["ml"]
    base_seed = int(ml_cfg.get("random_state", 42))
    repeat_seeds = sorted(set(int(v) for v in ml_cfg.get("repeat_seeds", [base_seed])))

    test_n = _compute_stratified_test_n(
        n_total=x.shape[0],
        n_strata=len(np.unique(strata)),
        requested_ratio=float(ml_cfg.get("test_size", 0.2)),
    )

    def make_builder(model_name: str):
        def _builder(seed: int):
            return _build_candidate_models(ml_cfg, seed=seed)[model_name]

        return _builder

    all_repeat_rows: list[dict] = []
    summaries = []
    for model_name in ["rf", "extratrees", "svc_rbf", "hard_vote_ensemble"]:
        rows, summary = _evaluate_repeats(
            model_name=model_name,
            model_builder=make_builder(model_name),
            x=x,
            y=y,
            strata=strata,
            test_n=test_n,
            seeds=repeat_seeds,
        )
        all_repeat_rows.extend(rows)
        summaries.append(summary)

    leaderboard = pd.DataFrame(summaries).sort_values(
        ["macro_f1_mean", "accuracy_mean", "macro_f1_std", "accuracy_std"],
        ascending=[False, False, True, True],
    )

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_repeat_rows).to_csv(reports_dir / "sota_repeats.csv", index=False)
    leaderboard.to_csv(reports_dir / "model_leaderboard.csv", index=False)

    best_model_name = str(leaderboard.iloc[0]["model"])
    best_builder = make_builder(best_model_name)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_n,
        random_state=base_seed,
        stratify=strata,
    )
    best_model = best_builder(base_seed)
    best_model.fit(x_train, y_train)
    pred = best_model.predict(x_test)

    holdout_metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }

    labels = sorted(np.unique(y))
    _plot_confusion(
        y_true=y_test,
        y_pred=pred,
        labels=labels,
        output_path=reports_dir / "confusion_matrix_sota.png",
        title=f"SOTA Candidate Confusion Matrix ({best_model_name})",
    )

    robust_rows = _robustness_curve(
        model=best_builder(base_seed),
        x=x,
        y=y,
        eps=features_df["eps"].to_numpy(dtype=float),
        model_cfg=ml_cfg,
        output_path=reports_dir / "robustness_sweep_sota.png",
    )

    re_rows = _leave_one_re_out(features_df, feature_cols, best_builder)

    _write_sota_summary(
        path=reports_dir / "sota_summary.md",
        best_model_name=best_model_name,
        best_seed=base_seed,
        holdout_metrics=holdout_metrics,
        leaderboard=leaderboard,
        robust_rows=robust_rows,
        re_rows=re_rows,
    )

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "sota.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": best_model,
                "model_name": best_model_name,
                "feature_columns": feature_cols,
                "labels": labels,
                "holdout_metrics": holdout_metrics,
                "leaderboard": leaderboard.to_dict(orient="records"),
                "config": cfg,
            },
            handle,
        )

    with (reports_dir / "sota_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_model": best_model_name,
                "holdout_metrics": holdout_metrics,
                "leaderboard_top": leaderboard.head(4).to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    logger.info(
        "SOTA candidate training done. best=%s acc=%.4f macro_f1=%.4f model=%s",
        best_model_name,
        holdout_metrics["accuracy"],
        holdout_metrics["macro_f1"],
        model_path,
    )


if __name__ == "__main__":
    main()
