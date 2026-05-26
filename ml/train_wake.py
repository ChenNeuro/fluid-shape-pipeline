from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.wake_common import (
    VARIANTS,
    accuracy_f1,
    build_label_maps,
    clip_params,
    compute_stratified_test_n,
    obstacle_iou_and_dice,
    render_targets,
    repeated_holdout_split,
    stratification_labels,
)
from ml.wake_reporting import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_variant_summary,
    write_wake_summary,
)
from ml.wake_splits import split_train_val as _split_train_val
from ml.wake_training import predict_wake_model as _predict
from ml.wake_training import save_model_pack as _save_model_pack
from ml.wake_training import set_seed as set_seed  # noqa: F401
from ml.wake_training import train_jepa_wake_model as _train_jepa_model  # noqa: F401
from ml.wake_training import train_resnet_wake_model as _train_model  # noqa: F401
from ml.wake_training import train_simsiam_wake_model as _train_simsiam_model  # noqa: F401
from ml.wake_training import train_smallcnn_wake_model as _train_smallcnn_model  # noqa: F401
from ml.wake_training import train_vit_wake_model as _train_vit_model  # noqa: F401
from ml.wake_training import train_wake_backbone
from sim.config import load_config
from sim.experiment import experiment_paths, write_run_manifest
from sim.logging_utils import setup_logger
from vision.wake_dataset import WakeBundle, load_wake_bundle, variant_tensor
from vision.wake_model import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multi-scale wake-field classifiers and export comparison reports"
    )
    parser.add_argument(
        "--config", default="configs/wake_field_450.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "mae_vit", "jepa", "smallcnn", "simsiam"],
        help="Encoder backbone architecture",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name for output subdirectory (default: backbone name)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment output directory. Defaults to runs/<config-name>.",
    )
    return parser.parse_args()


def _labels_for_indices(
    bundle: WakeBundle, label_maps, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shape_idx = np.asarray(
        [label_maps.shape_to_idx[value] for value in bundle.shapes[indices]],
        dtype=np.int64,
    )
    re_idx = np.asarray(
        [label_maps.re_to_idx[int(value)] for value in bundle.re_values[indices]],
        dtype=np.int64,
    )
    return shape_idx, re_idx


def _params_for_indices(bundle: WakeBundle, indices: np.ndarray) -> np.ndarray:
    if indices.shape[0] == 0:
        return np.array([]).reshape(0, 2).astype(np.float32)
    return np.stack([bundle.dy[indices], bundle.eps[indices]], axis=1).astype(np.float32)


def _train_for_indices(
    *,
    backbone: str,
    x_all: np.ndarray,
    bundle: WakeBundle,
    label_maps,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    cfg: dict,
    seed: int,
    device,
):
    shape_train_idx, re_train_idx = _labels_for_indices(bundle, label_maps, idx_train)
    shape_val_idx, re_val_idx = (
        _labels_for_indices(bundle, label_maps, idx_val)
        if idx_val.shape[0] > 0
        else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )

    return train_wake_backbone(
        backbone=backbone,
        x_train=x_all[idx_train],
        shape_train_idx=shape_train_idx,
        params_train=_params_for_indices(bundle, idx_train),
        re_train_idx=re_train_idx,
        x_val=x_all[idx_val],
        shape_val_idx=shape_val_idx,
        params_val=_params_for_indices(bundle, idx_val),
        re_val_idx=re_val_idx,
        cfg=cfg,
        seed=seed,
        n_shapes=len(label_maps.shape_to_idx),
        n_re_classes=len(label_maps.re_to_idx),
        device=device,
    )


def _evaluate_predictions(
    *,
    pred: dict[str, np.ndarray],
    bundle: WakeBundle,
    label_maps,
    idx_test: np.ndarray,
    cfg: dict,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    shape_pred_idx = np.argmax(pred["shape_logits"], axis=1)
    re_pred_idx = np.argmax(pred["re_logits"], axis=1)
    shape_pred = np.asarray(
        [label_maps.idx_to_shape[int(idx)] for idx in shape_pred_idx], dtype=object
    )
    dy_pred, eps_pred = clip_params(pred["params_pred"][:, 0], pred["params_pred"][:, 1], cfg)

    dy_true = bundle.dy[idx_test].astype(np.float32)
    eps_true = bundle.eps[idx_test].astype(np.float32)
    targets = render_targets(
        shapes=bundle.shapes[idx_test],
        dy_values=dy_true,
        eps_values=eps_true,
        cfg=cfg,
    )
    predictions = render_targets(
        shapes=shape_pred,
        dy_values=dy_pred,
        eps_values=eps_pred,
        cfg=cfg,
    )
    inv_metrics = obstacle_iou_and_dice(
        targets,
        predictions,
        threshold=float(cfg["reconstruction"].get("obstacle_threshold", 0.8)),
    )
    cls_metrics = accuracy_f1(bundle.shapes[idx_test], shape_pred)
    re_acc = float(
        np.mean(
            re_pred_idx
            == np.asarray(
                [label_maps.re_to_idx[int(value)] for value in bundle.re_values[idx_test]],
                dtype=int,
            )
        )
    )

    metrics = {
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "re_accuracy": re_acc,
        "dy_mae": float(np.mean(np.abs(dy_pred - dy_true))),
        "eps_mae": float(np.mean(np.abs(eps_pred - eps_true))),
        "inverse_iou": inv_metrics["iou_mean"],
        "inverse_dice": inv_metrics["dice_mean"],
    }
    return metrics, shape_pred, dy_pred, eps_pred


def _shape_labels(label_maps) -> list[str]:
    return [label_maps.idx_to_shape[idx] for idx in sorted(label_maps.idx_to_shape)]


def _re_values(label_maps) -> list[int]:
    return [label_maps.idx_to_re[idx] for idx in sorted(label_maps.idx_to_re)]


def _train_repeated_holdouts(
    *,
    args: argparse.Namespace,
    cfg: dict,
    bundle: WakeBundle,
    label_maps,
    strata: np.ndarray,
    repeat_seeds: list[int],
    export_seed: int,
    test_n: int,
    val_ratio: float,
    batch_size: int,
    device,
    models_dir,
    reports_dir,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, float]]]]:
    all_rows = []
    histories_for_plot: dict[str, list[dict[str, float]]] = {}
    main_variant = "dist_multi_4ch"
    single_variant = "dist_single_4ch"

    for variant_name, spec in VARIANTS.items():
        x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])

        for seed in repeat_seeds:
            idx_train, idx_test = repeated_holdout_split(strata, test_n=test_n, seed=seed)
            idx_train_real, idx_val = _split_train_val(idx_train, strata, val_ratio, seed)

            model, history = _train_for_indices(
                backbone=args.backbone,
                x_all=x_all,
                bundle=bundle,
                label_maps=label_maps,
                idx_train=idx_train_real,
                idx_val=idx_val,
                cfg=cfg,
                seed=seed,
                device=device,
            )
            if seed == export_seed:
                histories_for_plot[variant_name] = history

            pred = _predict(model, x_all[idx_test], batch_size=batch_size, device=device)
            metrics, shape_pred, _dy_pred, _eps_pred = _evaluate_predictions(
                pred=pred, bundle=bundle, label_maps=label_maps, idx_test=idx_test, cfg=cfg
            )
            all_rows.append(
                {
                    "variant": variant_name,
                    "description": spec["description"],
                    "seed": int(seed),
                    "train_size": int(idx_train_real.shape[0]),
                    "val_size": int(idx_val.shape[0]),
                    "test_size": int(idx_test.shape[0]),
                    **metrics,
                }
            )

            if seed == export_seed and variant_name in {single_variant, main_variant}:
                output_name = (
                    "wake_field_single.pt"
                    if variant_name == single_variant
                    else "wake_field_main.pt"
                )
                _save_model_pack(
                    output_path=models_dir / output_name,
                    model=model.cpu(),
                    model_type=args.backbone,
                    variant_name=variant_name,
                    x_shape=x_all.shape,
                    shape_labels=_shape_labels(label_maps),
                    re_values=_re_values(label_maps),
                    test_case_ids=bundle.case_ids[idx_test].tolist(),
                    cfg=cfg,
                    seed=seed,
                )
                model = model.to(device)

            if seed == export_seed and variant_name == main_variant:
                plot_confusion_matrix(
                    y_true=bundle.shapes[idx_test],
                    y_pred=shape_pred,
                    labels=_shape_labels(label_maps),
                    output_path=reports_dir / "wake_field_confusion_matrix.png",
                )

    holdout_df = pd.DataFrame(all_rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    return holdout_df, histories_for_plot


def _summarize_holdouts(holdout_df: pd.DataFrame) -> pd.DataFrame:
    return (
        holdout_df.groupby(["variant", "description"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            re_accuracy_mean=("re_accuracy", "mean"),
            dy_mae_mean=("dy_mae", "mean"),
            eps_mae_mean=("eps_mae", "mean"),
            inverse_iou_mean=("inverse_iou", "mean"),
            inverse_dice_mean=("inverse_dice", "mean"),
        )
        .fillna(0.0)
    )


def _leave_one_re_out(
    *,
    args: argparse.Namespace,
    cfg: dict,
    bundle: WakeBundle,
    label_maps,
    strata: np.ndarray,
    val_ratio: float,
    batch_size: int,
    device,
) -> pd.DataFrame:
    main_spec = VARIANTS["dist_multi_4ch"]
    x_main = variant_tensor(bundle, scales=main_spec["scales"], channels=main_spec["channels"])
    leave_rows = []

    for re_test in sorted(np.unique(bundle.re_values)):
        idx_train = np.where(bundle.re_values != re_test)[0]
        idx_test = np.where(bundle.re_values == re_test)[0]
        idx_train_real, idx_val = _split_train_val(
            idx_train, strata, val_ratio, seed=1000 + int(re_test)
        )
        model, _ = _train_for_indices(
            backbone=args.backbone,
            x_all=x_main,
            bundle=bundle,
            label_maps=label_maps,
            idx_train=idx_train_real,
            idx_val=idx_val,
            cfg=cfg,
            seed=1000 + int(re_test),
            device=device,
        )
        pred = _predict(model, x_main[idx_test], batch_size=batch_size, device=device)
        metrics, _shape_pred, _dy_pred, _eps_pred = _evaluate_predictions(
            pred=pred, bundle=bundle, label_maps=label_maps, idx_test=idx_test, cfg=cfg
        )
        leave_rows.append(
            {
                "Re_test": int(re_test),
                "n_test": int(idx_test.shape[0]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "dy_mae": metrics["dy_mae"],
                "eps_mae": metrics["eps_mae"],
                "inverse_iou": metrics["inverse_iou"],
                "inverse_dice": metrics["inverse_dice"],
            }
        )

    return pd.DataFrame(leave_rows).sort_values("Re_test").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = experiment_paths(cfg, config_path=args.config, run_dir=args.run_dir)
    run_name = args.run_name or args.backbone
    write_run_manifest(
        paths=paths,
        cfg=cfg,
        config_path=args.config,
        stage="train_wake",
        extra={"backbone": args.backbone, "run_name": run_name},
    )
    logger = setup_logger("train_wake", paths.logs_dir / f"train_wake_{run_name}.log")

    bundle = load_wake_bundle(paths.wake_fields_dir)
    label_maps = build_label_maps(bundle)
    strata = stratification_labels(bundle)
    ml_cfg = cfg["ml"]
    test_n = compute_stratified_test_n(
        bundle.case_ids.shape[0], len(np.unique(strata)), float(ml_cfg.get("test_size", 0.2))
    )
    repeat_seeds = [int(seed) for seed in ml_cfg.get("repeat_seeds", [42])]
    export_seed = int(cfg.get("vision", {}).get("training", {}).get("export_seed", repeat_seeds[0]))
    if export_seed not in repeat_seeds:
        export_seed = repeat_seeds[0]
    batch_size = int(cfg.get("vision", {}).get("training", {}).get("batch_size", 16))
    device = select_device()
    val_ratio = float(cfg.get("vision", {}).get("training", {}).get("val_ratio", 0.0))

    models_dir = paths.models_dir / run_name
    reports_dir = paths.reports_dir / run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training wake-field backbone=%s run=%s", args.backbone, run_name)
    holdout_df, histories_for_plot = _train_repeated_holdouts(
        args=args,
        cfg=cfg,
        bundle=bundle,
        label_maps=label_maps,
        strata=strata,
        repeat_seeds=repeat_seeds,
        export_seed=export_seed,
        test_n=test_n,
        val_ratio=val_ratio,
        batch_size=batch_size,
        device=device,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )
    holdout_df.to_csv(reports_dir / "wake_field_holdout_repeats.csv", index=False)

    summary_df = _summarize_holdouts(holdout_df)
    summary_df.to_csv(reports_dir / "wake_field_variant_summary.csv", index=False)
    plot_variant_summary(summary_df, reports_dir / "wake_field_variant_comparison.png")
    plot_training_curves(
        histories_for_plot, reports_dir / "wake_field_training_curves.png", export_seed=export_seed
    )

    leave_one_re_df = _leave_one_re_out(
        args=args,
        cfg=cfg,
        bundle=bundle,
        label_maps=label_maps,
        strata=strata,
        val_ratio=val_ratio,
        batch_size=batch_size,
        device=device,
    )
    leave_one_re_df.to_csv(reports_dir / "wake_field_leave_one_re_out.csv", index=False)
    write_wake_summary(
        output_path=reports_dir / "wake_field_summary.md",
        summary_df=summary_df,
        leave_one_re_df=leave_one_re_df,
        main_variant="dist_multi_4ch",
        single_variant="dist_single_4ch",
    )

    selection_payload = {
        "main_variant": "dist_multi_4ch",
        "speed_variant": None,
        "device": str(device),
        "repeat_seeds": repeat_seeds,
        "test_size": int(test_n),
        "run_dir": str(paths.run_dir),
        "wake_fields_dir": str(paths.wake_fields_dir),
        "models_dir": str(models_dir),
        "reports_dir": str(reports_dir),
    }
    (reports_dir / "wake_field_selection.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wake-field training complete. main_variant=%s summary=%s",
        "dist_multi_4ch",
        reports_dir / "wake_field_summary.md",
    )


if __name__ == "__main__":
    main()
