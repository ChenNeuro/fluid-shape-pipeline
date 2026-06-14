from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from ml.wake_common import (
    VARIANTS,
    accuracy_f1,
    compute_stratified_test_n,
    repeated_holdout_split,
    stratification_labels,
)
from sim.config import load_config
from sim.experiment import experiment_paths, write_run_manifest
from sim.logging_utils import setup_logger
from vision.wake_dataset import WakeBundle, load_wake_bundle, variant_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit shortcut/leakage baselines for wake-field experiments"
    )
    parser.add_argument(
        "--config", default="configs/wake_field_450.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment output directory. Defaults to runs/<config-name>.",
    )
    parser.add_argument(
        "--variant",
        default="dist_multi_4ch",
        choices=sorted(VARIANTS),
        help="Wake-field variant used for crop-stat shortcut features.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="ExtraTrees estimators for learned shortcut baselines.",
    )
    return parser.parse_args()


def _sample_idx(case_id: str) -> int:
    match = re.search(r"_p(\d+)$", case_id)
    return int(match.group(1)) if match else -1


def nuisance_features(bundle: WakeBundle) -> np.ndarray:
    sample_idx = np.asarray([_sample_idx(case_id) for case_id in bundle.case_ids], dtype=float)
    return np.column_stack(
        [
            bundle.re_values.astype(float),
            bundle.dy.astype(float),
            bundle.eps.astype(float),
            sample_idx,
        ]
    )


def crop_stat_features(bundle: WakeBundle, *, variant_name: str) -> np.ndarray:
    spec = VARIANTS[variant_name]
    x = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])
    stats = [
        x.mean(axis=(-2, -1)),
        x.std(axis=(-2, -1)),
        x.min(axis=(-2, -1)),
        x.max(axis=(-2, -1)),
        np.quantile(x, 0.10, axis=(-2, -1)),
        np.quantile(x, 0.90, axis=(-2, -1)),
    ]
    return np.concatenate([item.reshape(item.shape[0], -1) for item in stats], axis=1)


def case_id_prefix_predictions(case_ids: np.ndarray) -> np.ndarray:
    return np.asarray([str(case_id).split("_Re", 1)[0] for case_id in case_ids], dtype=object)


def _fit_predict_extra_trees(
    *,
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x[idx_train], y[idx_train])
    return np.asarray(model.predict(x[idx_test]))


def evaluate_shortcut_baselines(
    *,
    bundle: WakeBundle,
    cfg: dict,
    variant_name: str,
    n_estimators: int,
) -> pd.DataFrame:
    y = bundle.shapes.astype(object)
    strata = stratification_labels(bundle)
    test_n = compute_stratified_test_n(
        bundle.case_ids.shape[0],
        len(np.unique(strata)),
        float(cfg["ml"].get("test_size", 0.2)),
    )
    repeat_seeds = [int(seed) for seed in cfg["ml"].get("repeat_seeds", [42])]
    features = {
        "nuisance_metadata": nuisance_features(bundle),
        "crop_low_order_stats": crop_stat_features(bundle, variant_name=variant_name),
    }

    rows = []
    for seed in repeat_seeds:
        idx_train, idx_test = repeated_holdout_split(strata, test_n=test_n, seed=seed)

        case_id_pred = case_id_prefix_predictions(bundle.case_ids[idx_test])
        case_id_metrics = accuracy_f1(y[idx_test], case_id_pred)
        rows.append(
            {
                "baseline": "case_id_prefix_oracle",
                "seed": int(seed),
                "train_size": int(idx_train.shape[0]),
                "test_size": int(idx_test.shape[0]),
                **case_id_metrics,
                "notes": "case_id encodes shape label; never use as a model feature",
            }
        )

        for baseline, x in features.items():
            pred = _fit_predict_extra_trees(
                x=x,
                y=y,
                idx_train=idx_train,
                idx_test=idx_test,
                seed=seed,
                n_estimators=n_estimators,
            )
            metrics = accuracy_f1(y[idx_test], pred)
            rows.append(
                {
                    "baseline": baseline,
                    "seed": int(seed),
                    "train_size": int(idx_train.shape[0]),
                    "test_size": int(idx_test.shape[0]),
                    **metrics,
                    "notes": "",
                }
            )

    return pd.DataFrame(rows).sort_values(["baseline", "seed"]).reset_index(drop=True)


def write_summary(*, output_path: Path, audit_df: pd.DataFrame, variant_name: str) -> None:
    summary = (
        audit_df.groupby("baseline", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_repeats=("seed", "count"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    lines = [
        "# Wake-Field Leakage Audit",
        "",
        f"- Crop-stat variant: `{variant_name}`",
        "- `case_id_prefix_oracle` is expected to be perfect because case IDs encode the label.",
        "- `nuisance_metadata` uses only Re, dy, eps, and perturbation index.",
        (
            "- `crop_low_order_stats` uses per-crop/channel low-order statistics, "
            "not CNN spatial structure."
        ),
        "",
        "## Summary",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['baseline']}: macroF1={row['macro_f1_mean']:.4f}+/-{row['macro_f1_std']:.4f}, "
            f"acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f}, n={int(row['n_repeats'])}"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            (
                "If nuisance metadata approaches the main model score, the split/design "
                "likely leaks shape through non-visual variables."
            ),
            (
                "If crop low-order stats approaches the main model score, claims about "
                "learned spatial wake structure need stronger ablations."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = experiment_paths(cfg, config_path=args.config, run_dir=args.run_dir)
    write_run_manifest(
        paths=paths,
        cfg=cfg,
        config_path=args.config,
        stage="audit_wake_leakage",
        extra={"variant": args.variant, "n_estimators": args.n_estimators},
    )
    logger = setup_logger("audit_wake_leakage", paths.logs_dir / "audit_wake_leakage.log")

    bundle = load_wake_bundle(paths.wake_fields_dir)
    audit_df = evaluate_shortcut_baselines(
        bundle=bundle,
        cfg=cfg,
        variant_name=args.variant,
        n_estimators=int(args.n_estimators),
    )
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.reports_dir / "wake_leakage_audit.csv"
    md_path = paths.reports_dir / "wake_leakage_audit.md"
    audit_df.to_csv(csv_path, index=False)
    write_summary(output_path=md_path, audit_df=audit_df, variant_name=args.variant)
    logger.info("Wake leakage audit complete: %s", md_path)


if __name__ == "__main__":
    main()
