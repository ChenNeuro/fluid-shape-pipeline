from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

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
from sim.config import load_config, repo_root
from sim.logging_utils import setup_logger
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleWakeNet, compute_multitask_loss, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-scale wake-field classifiers and export comparison reports")
    parser.add_argument("--config", default="configs/wake_field_450.yaml", help="Path to YAML config")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "mae_vit"],
        help="Encoder backbone architecture",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_loader(x: np.ndarray, shape_idx: np.ndarray, params: np.ndarray, re_idx: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(shape_idx).long(),
        torch.from_numpy(params).float(),
        torch.from_numpy(re_idx).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _predict(model: torch.nn.Module, x: np.ndarray, *, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    dataset = TensorDataset(torch.from_numpy(x).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    shape_logits = []
    params_pred = []
    re_logits = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs = model(batch_x.to(device))
            shape_logits.append(outputs["shape_logits"].detach().cpu().numpy())
            params_pred.append(outputs["params_pred"].detach().cpu().numpy())
            re_logits.append(outputs["re_logits"].detach().cpu().numpy())

    return {
        "shape_logits": np.concatenate(shape_logits, axis=0),
        "params_pred": np.concatenate(params_pred, axis=0),
        "re_logits": np.concatenate(re_logits, axis=0),
    }


def _train_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[MultiScaleWakeNet, list[dict[str, float]]]:
    train_cfg = cfg.get("vision", {}).get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 8))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    fusion_hidden = int(train_cfg.get("fusion_hidden", 256))
    dropout = float(train_cfg.get("dropout", 0.15))

    set_seed(seed)
    model = MultiScaleWakeNet(
        n_scales=int(x_train.shape[1]),
        in_channels=int(x_train.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        fusion_hidden=fusion_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _tensor_loader(x_train, shape_train_idx, params_train, re_train_idx, batch_size=batch_size)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        loss_shape = 0.0
        loss_params = 0.0
        loss_re = 0.0
        n_items = 0

        for batch_x, batch_shape, batch_params, batch_re in loader:
            batch_x = batch_x.to(device)
            batch_shape = batch_shape.to(device)
            batch_params = batch_params.to(device)
            batch_re = batch_re.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss, loss_parts = compute_multitask_loss(
                outputs,
                shape_target=batch_shape,
                param_target=batch_params,
                re_target=batch_re,
            )
            loss.backward()
            optimizer.step()

            batch_n = int(batch_x.shape[0])
            n_items += batch_n
            loss_total += loss_parts["loss_total"] * batch_n
            loss_shape += loss_parts["loss_shape"] * batch_n
            loss_params += loss_parts["loss_params"] * batch_n
            loss_re += loss_parts["loss_re"] * batch_n

        history.append(
            {
                "epoch": epoch + 1,
                "loss_total": loss_total / max(n_items, 1),
                "loss_shape": loss_shape / max(n_items, 1),
                "loss_params": loss_params / max(n_items, 1),
                "loss_re": loss_re / max(n_items, 1),
            }
        )

    return model, history


def _train_vit_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from vision.mae_vit_model import MultiScaleViTWakeNet

    vit_cfg = cfg.get("vision", {}).get("mae_vit", {})
    batch_size = int(vit_cfg.get("batch_size", 8))
    phase1_epochs = int(vit_cfg.get("phase1_epochs", 12))
    phase2_epochs = int(vit_cfg.get("phase2_epochs", 13))
    phase1_lr = float(vit_cfg.get("phase1_lr", 2e-3))
    phase2_base_lr = float(vit_cfg.get("phase2_base_lr", 5e-5))
    llrd_decay = float(vit_cfg.get("llrd_decay", 0.85))
    proj_dim = int(vit_cfg.get("proj_dim", 512))
    fusion_hidden = int(vit_cfg.get("fusion_hidden", 512))
    dropout = float(vit_cfg.get("dropout", 0.2))

    set_seed(seed)
    model = MultiScaleViTWakeNet(
        n_scales=int(x_train.shape[1]),
        in_channels=int(x_train.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        proj_dim=proj_dim,
        fusion_hidden=fusion_hidden,
        dropout=dropout,
        pretrained=True,
    ).to(device)

    loader = _tensor_loader(x_train, shape_train_idx, params_train, re_train_idx, batch_size=batch_size)
    history: list[dict[str, float]] = []

    model.freeze_backbone()
    optimizer_p1 = torch.optim.AdamW(
        list(model.scale_proj.parameters())
        + list(model.fusion.parameters())
        + list(model.shape_head.parameters())
        + list(model.params_head.parameters())
        + list(model.re_head.parameters()),
        lr=phase1_lr,
        weight_decay=1e-4,
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs)
    for epoch in range(phase1_epochs):
        model.train()
        loss_total = 0.0
        n_items = 0
        for batch_x, batch_shape, batch_params, batch_re in loader:
            batch_x = batch_x.to(device)
            batch_shape = batch_shape.to(device)
            batch_params = batch_params.to(device)
            batch_re = batch_re.to(device)

            optimizer_p1.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss, _ = compute_multitask_loss(
                outputs,
                shape_target=batch_shape,
                param_target=batch_params,
                re_target=batch_re,
            )
            loss.backward()
            optimizer_p1.step()

            batch_n = int(batch_x.shape[0])
            n_items += batch_n
            loss_total += float(loss.detach().cpu()) * batch_n

        scheduler_p1.step()
        history.append(
            {
                "epoch": epoch + 1,
                "phase": 1,
                "loss_total": loss_total / max(n_items, 1),
            }
        )

    llrd_groups = model.unfreeze_with_llrd(base_lr=phase2_base_lr, llrd_decay=llrd_decay)
    optimizer_p2 = torch.optim.AdamW(llrd_groups, lr=phase2_base_lr)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=phase2_epochs)

    for epoch in range(phase2_epochs):
        model.train()
        loss_total = 0.0
        n_items = 0
        for batch_x, batch_shape, batch_params, batch_re in loader:
            batch_x = batch_x.to(device)
            batch_shape = batch_shape.to(device)
            batch_params = batch_params.to(device)
            batch_re = batch_re.to(device)

            optimizer_p2.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss, _ = compute_multitask_loss(
                outputs,
                shape_target=batch_shape,
                param_target=batch_params,
                re_target=batch_re,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_p2.step()

            batch_n = int(batch_x.shape[0])
            n_items += batch_n
            loss_total += float(loss.detach().cpu()) * batch_n

        scheduler_p2.step()
        history.append(
            {
                "epoch": phase1_epochs + epoch + 1,
                "phase": 2,
                "loss_total": loss_total / max(n_items, 1),
            }
        )

    return model, history


def _save_model_pack(
    *,
    output_path: Path,
    model: torch.nn.Module,
    model_type: str,
    variant_name: str,
    x_shape: tuple[int, ...],
    shape_labels: list[str],
    re_values: list[int],
    test_case_ids: list[str],
    cfg: dict,
    seed: int,
) -> None:
    if model_type == "resnet18":
        model_kwargs = {
            "n_scales": int(x_shape[1]),
            "in_channels": int(x_shape[2]),
            "n_shapes": len(shape_labels),
            "n_re_classes": len(re_values),
            "fusion_hidden": int(cfg.get("vision", {}).get("training", {}).get("fusion_hidden", 256)),
            "dropout": float(cfg.get("vision", {}).get("training", {}).get("dropout", 0.15)),
        }
    else:
        model_kwargs = {
            "n_scales": int(x_shape[1]),
            "in_channels": int(x_shape[2]),
            "n_shapes": len(shape_labels),
            "n_re_classes": len(re_values),
            "proj_dim": int(cfg.get("vision", {}).get("mae_vit", {}).get("proj_dim", 512)),
            "fusion_hidden": int(cfg.get("vision", {}).get("mae_vit", {}).get("fusion_hidden", 512)),
            "dropout": float(cfg.get("vision", {}).get("mae_vit", {}).get("dropout", 0.2)),
        }
    payload = {
        "model_type": model_type,
        "variant_name": variant_name,
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "shape_labels": shape_labels,
        "re_values": re_values,
        "test_case_ids": test_case_ids,
        "fit_seed": int(seed),
        "config_snapshot": cfg,
    }
    torch.save(payload, output_path)


def _plot_variant_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_df.sort_values("macro_f1_mean").reset_index(drop=True)
    x = np.arange(ordered.shape[0], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x, ordered["macro_f1_mean"], yerr=ordered["macro_f1_std"], color="#0ea5e9", alpha=0.85)
    ax.set_xticks(x, ordered["variant"], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Wake-Field Variant Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_training_curves(
    histories: dict[str, list[dict[str, float]]],
    output_path: Path,
    *,
    export_seed: int,
) -> None:
    if not histories:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for variant_name, history in histories.items():
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        losses = [row["loss_total"] for row in history]
        ax.plot(epochs, losses, marker="o", label=variant_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(f"Wake-Field Training Curves (seed={export_seed})")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Wake-Field Main Variant Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(
    *,
    output_path: Path,
    summary_df: pd.DataFrame,
    leave_one_re_df: pd.DataFrame,
    main_variant: str,
    single_variant: str,
) -> None:
    summary_sorted = summary_df.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    main_row = summary_sorted.loc[summary_sorted["variant"] == main_variant].iloc[0]
    single_row = summary_sorted.loc[summary_sorted["variant"] == single_variant].iloc[0]

    lines = [
        "# Wake-Field Training Summary",
        "",
        f"- Main variant: `{main_variant}`",
        f"- Main repeated holdout: acc={main_row['accuracy_mean']:.4f}+/-{main_row['accuracy_std']:.4f}, macroF1={main_row['macro_f1_mean']:.4f}+/-{main_row['macro_f1_std']:.4f}",
        f"- Single-scale (`{single_variant}`) vs multi-scale macroF1: {single_row['macro_f1_mean']:.4f} -> {main_row['macro_f1_mean']:.4f}",
        "",
        "## Repeated Holdout Comparison",
    ]

    for _, row in summary_sorted.iterrows():
        lines.append(
            f"- {row['variant']}: acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f}, "
            f"macroF1={row['macro_f1_mean']:.4f}+/-{row['macro_f1_std']:.4f}, "
            f"dy_MAE={row['dy_mae_mean']:.5f}, eps_MAE={row['eps_mae_mean']:.5f}, "
            f"IoU={row['inverse_iou_mean']:.4f}, Dice={row['inverse_dice_mean']:.4f}"
        )

    lines.append("")
    lines.append("## Leave-One-Re-Out (Main Variant)")
    for _, row in leave_one_re_df.iterrows():
        lines.append(
            f"- Re={int(row['Re_test'])}: acc={row['accuracy']:.4f}, macroF1={row['macro_f1']:.4f}, "
            f"dy_MAE={row['dy_mae']:.5f}, eps_MAE={row['eps_mae']:.5f}, IoU={row['inverse_iou']:.4f}, Dice={row['inverse_dice']:.4f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("train_wake", root / "logs" / "train_wake.log")

    bundle = load_wake_bundle()
    label_maps = build_label_maps(bundle)
    strata = stratification_labels(bundle)
    ml_cfg = cfg["ml"]
    test_n = compute_stratified_test_n(bundle.case_ids.shape[0], len(np.unique(strata)), float(ml_cfg.get("test_size", 0.2)))
    repeat_seeds = [int(seed) for seed in ml_cfg.get("repeat_seeds", [42])]
    export_seed = int(cfg.get("vision", {}).get("training", {}).get("export_seed", repeat_seeds[0]))
    if export_seed not in repeat_seeds:
        export_seed = repeat_seeds[0]
    batch_size = int(cfg.get("vision", {}).get("training", {}).get("batch_size", 16))
    device = select_device()

    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    histories_for_plot: dict[str, list[dict[str, float]]] = {}
    main_variant = "dist_multi_4ch"
    single_variant = "dist_single_4ch"

    for variant_name, spec in VARIANTS.items():
        logger.info("Training wake-field variant %s with backbone=%s", variant_name, args.backbone)
        x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])

        for seed in repeat_seeds:
            idx_train, idx_test = repeated_holdout_split(strata, test_n=test_n, seed=seed)
            shape_train_idx = np.asarray([label_maps.shape_to_idx[value] for value in bundle.shapes[idx_train]], dtype=np.int64)
            re_train_idx = np.asarray([label_maps.re_to_idx[int(value)] for value in bundle.re_values[idx_train]], dtype=np.int64)
            params_train = np.stack([bundle.dy[idx_train], bundle.eps[idx_train]], axis=1).astype(np.float32)

            if args.backbone == "mae_vit":
                model, history = _train_vit_model(
                    x_train=x_all[idx_train],
                    shape_train_idx=shape_train_idx,
                    params_train=params_train,
                    re_train_idx=re_train_idx,
                    cfg=cfg,
                    seed=seed,
                    n_shapes=len(label_maps.shape_to_idx),
                    n_re_classes=len(label_maps.re_to_idx),
                    device=device,
                )
            else:
                model, history = _train_model(
                    x_train=x_all[idx_train],
                    shape_train_idx=shape_train_idx,
                    params_train=params_train,
                    re_train_idx=re_train_idx,
                    cfg=cfg,
                    seed=seed,
                    n_shapes=len(label_maps.shape_to_idx),
                    n_re_classes=len(label_maps.re_to_idx),
                    device=device,
                )

            if seed == export_seed:
                histories_for_plot[variant_name] = history

            pred = _predict(model, x_all[idx_test], batch_size=batch_size, device=device)
            shape_pred_idx = np.argmax(pred["shape_logits"], axis=1)
            re_pred_idx = np.argmax(pred["re_logits"], axis=1)
            shape_pred = np.asarray([label_maps.idx_to_shape[int(idx)] for idx in shape_pred_idx], dtype=object)

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
                    == np.asarray([label_maps.re_to_idx[int(value)] for value in bundle.re_values[idx_test]], dtype=int)
                )
            )

            row = {
                "variant": variant_name,
                "description": spec["description"],
                "seed": int(seed),
                "train_size": int(idx_train.shape[0]),
                "test_size": int(idx_test.shape[0]),
                "accuracy": cls_metrics["accuracy"],
                "macro_f1": cls_metrics["macro_f1"],
                "re_accuracy": re_acc,
                "dy_mae": float(np.mean(np.abs(dy_pred - dy_true))),
                "eps_mae": float(np.mean(np.abs(eps_pred - eps_true))),
                "inverse_iou": inv_metrics["iou_mean"],
                "inverse_dice": inv_metrics["dice_mean"],
            }
            all_rows.append(row)

            if seed == export_seed and variant_name == single_variant:
                _save_model_pack(
                    output_path=models_dir / "wake_field_single.pt",
                    model=model.cpu(),
                    model_type=args.backbone,
                    variant_name=variant_name,
                    x_shape=x_all.shape,
                    shape_labels=[label_maps.idx_to_shape[idx] for idx in sorted(label_maps.idx_to_shape)],
                    re_values=[label_maps.idx_to_re[idx] for idx in sorted(label_maps.idx_to_re)],
                    test_case_ids=bundle.case_ids[idx_test].tolist(),
                    cfg=cfg,
                    seed=seed,
                )
                model = model.to(device)

            if seed == export_seed and variant_name == main_variant:
                _save_model_pack(
                    output_path=models_dir / "wake_field_main.pt",
                    model=model.cpu(),
                    model_type=args.backbone,
                    variant_name=variant_name,
                    x_shape=x_all.shape,
                    shape_labels=[label_maps.idx_to_shape[idx] for idx in sorted(label_maps.idx_to_shape)],
                    re_values=[label_maps.idx_to_re[idx] for idx in sorted(label_maps.idx_to_re)],
                    test_case_ids=bundle.case_ids[idx_test].tolist(),
                    cfg=cfg,
                    seed=seed,
                )
                model = model.to(device)
                _plot_confusion_matrix(
                    y_true=bundle.shapes[idx_test],
                    y_pred=shape_pred,
                    labels=[label_maps.idx_to_shape[idx] for idx in sorted(label_maps.idx_to_shape)],
                    output_path=reports_dir / "wake_field_confusion_matrix.png",
                )

            logger.info(
                "Variant %s seed=%d done. acc=%.4f macro_f1=%.4f dy_mae=%.5f eps_mae=%.5f",
                variant_name,
                seed,
                row["accuracy"],
                row["macro_f1"],
                row["dy_mae"],
                row["eps_mae"],
            )

    holdout_df = pd.DataFrame(all_rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    holdout_df.to_csv(reports_dir / "wake_field_holdout_repeats.csv", index=False)

    summary_df = (
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
    summary_df.to_csv(reports_dir / "wake_field_variant_summary.csv", index=False)

    _plot_variant_summary(summary_df, reports_dir / "wake_field_variant_comparison.png")
    _plot_training_curves(histories_for_plot, reports_dir / "wake_field_training_curves.png", export_seed=export_seed)

    main_spec = VARIANTS[main_variant]
    x_main = variant_tensor(bundle, scales=main_spec["scales"], channels=main_spec["channels"])
    leave_rows = []
    for re_test in sorted(np.unique(bundle.re_values)):
        idx_train = np.where(bundle.re_values != re_test)[0]
        idx_test = np.where(bundle.re_values == re_test)[0]

        if args.backbone == "mae_vit":
            model, _ = _train_vit_model(
                x_train=x_main[idx_train],
                shape_train_idx=np.asarray([label_maps.shape_to_idx[value] for value in bundle.shapes[idx_train]], dtype=np.int64),
                params_train=np.stack([bundle.dy[idx_train], bundle.eps[idx_train]], axis=1).astype(np.float32),
                re_train_idx=np.asarray([label_maps.re_to_idx[int(value)] for value in bundle.re_values[idx_train]], dtype=np.int64),
                cfg=cfg,
                seed=1000 + int(re_test),
                n_shapes=len(label_maps.shape_to_idx),
                n_re_classes=len(label_maps.re_to_idx),
                device=device,
            )
        else:
            model, _ = _train_model(
                x_train=x_main[idx_train],
                shape_train_idx=np.asarray([label_maps.shape_to_idx[value] for value in bundle.shapes[idx_train]], dtype=np.int64),
                params_train=np.stack([bundle.dy[idx_train], bundle.eps[idx_train]], axis=1).astype(np.float32),
                re_train_idx=np.asarray([label_maps.re_to_idx[int(value)] for value in bundle.re_values[idx_train]], dtype=np.int64),
                cfg=cfg,
                seed=1000 + int(re_test),
                n_shapes=len(label_maps.shape_to_idx),
                n_re_classes=len(label_maps.re_to_idx),
                device=device,
            )
        pred = _predict(model, x_main[idx_test], batch_size=batch_size, device=device)
        shape_pred = np.asarray([label_maps.idx_to_shape[int(idx)] for idx in np.argmax(pred["shape_logits"], axis=1)], dtype=object)
        dy_pred, eps_pred = clip_params(pred["params_pred"][:, 0], pred["params_pred"][:, 1], cfg)
        targets = render_targets(shapes=bundle.shapes[idx_test], dy_values=bundle.dy[idx_test], eps_values=bundle.eps[idx_test], cfg=cfg)
        predictions = render_targets(shapes=shape_pred, dy_values=dy_pred, eps_values=eps_pred, cfg=cfg)
        inv_metrics = obstacle_iou_and_dice(targets, predictions, threshold=float(cfg["reconstruction"].get("obstacle_threshold", 0.8)))
        cls_metrics = accuracy_f1(bundle.shapes[idx_test], shape_pred)

        leave_rows.append(
            {
                "Re_test": int(re_test),
                "n_test": int(idx_test.shape[0]),
                "accuracy": cls_metrics["accuracy"],
                "macro_f1": cls_metrics["macro_f1"],
                "dy_mae": float(np.mean(np.abs(dy_pred - bundle.dy[idx_test]))),
                "eps_mae": float(np.mean(np.abs(eps_pred - bundle.eps[idx_test]))),
                "inverse_iou": inv_metrics["iou_mean"],
                "inverse_dice": inv_metrics["dice_mean"],
            }
        )
        logger.info(
            "Leave-one-Re-out main variant Re=%d done. acc=%.4f macro_f1=%.4f",
            int(re_test),
            cls_metrics["accuracy"],
            cls_metrics["macro_f1"],
        )

    leave_one_re_df = pd.DataFrame(leave_rows).sort_values("Re_test").reset_index(drop=True)
    leave_one_re_df.to_csv(reports_dir / "wake_field_leave_one_re_out.csv", index=False)

    _write_summary(
        output_path=reports_dir / "wake_field_summary.md",
        summary_df=summary_df,
        leave_one_re_df=leave_one_re_df,
        main_variant=main_variant,
        single_variant=single_variant,
    )

    selection_payload = {
        "main_variant": main_variant,
        "speed_variant": None,
        "device": str(device),
        "repeat_seeds": repeat_seeds,
        "test_size": int(test_n),
    }
    (reports_dir / "wake_field_selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    logger.info(
        "Wake-field training complete. main_variant=%s summary=%s",
        main_variant,
        reports_dir / "wake_field_summary.md",
    )


if __name__ == "__main__":
    main()
