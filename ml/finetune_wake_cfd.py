from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from ml.wake_common import VARIANTS
from ml.wake_training import (
    _run_supervised_epochs,
    predict_wake_model,
    set_seed,
    tensor_loader,
    train_wake_backbone,
)
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleJEPAModel, MultiScaleWakeNet, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a wake model on fixed-split CFD wake fields"
    )
    parser.add_argument(
        "--synthetic-model",
        default=None,
        help="Optional path to synthetic wake_field_main.pt. Omit for CFD-only training.",
    )
    parser.add_argument(
        "--cfd-run-dir", required=True, help="CFD run directory containing data/wake_fields"
    )
    parser.add_argument(
        "--output-run-dir", required=True, help="Output directory for CFD fine-tune reports/model"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Fine-tune learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "smallcnn", "jepa"],
        default="resnet18",
        help="Backbone for CFD-only training. Synthetic checkpoints remain resnet18-only.",
    )
    parser.add_argument(
        "--variant",
        default="auto",
        help="Wake variant to use for CFD-only training. Use auto for distD_multi_4ch.",
    )
    parser.add_argument("--fusion-hidden", type=int, default=256, help="CFD-only hidden width")
    parser.add_argument("--dropout", type=float, default=0.15, help="CFD-only dropout")
    parser.add_argument(
        "--encoder-norm",
        choices=["batch", "group", "identity"],
        default="group",
        help="Normalization for CFD-only CNN/JEPA encoder.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=40, help="JEPA pretrain epochs")
    parser.add_argument("--pretrain-lr", type=float, default=0.001, help="JEPA pretrain LR")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="JEPA mask ratio")
    parser.add_argument("--shape-weight", type=float, default=1.0, help="Shape loss weight")
    parser.add_argument("--params-weight", type=float, default=0.1, help="dy/eps loss weight")
    parser.add_argument("--re-weight", type=float, default=0.1, help="Re loss weight")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Training noise std")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Use all CFD rows for training and leave no internal test split.",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Also fine-tune the encoder. Default freezes encoder for small CFD sets.",
    )
    return parser.parse_args()


def _instantiate_model(pack: dict, device: torch.device) -> torch.nn.Module:
    model_type = str(pack.get("model_type", "resnet18"))
    if model_type != "resnet18":
        raise RuntimeError(
            f"CFD fine-tune currently supports resnet18 checkpoints only, got {model_type}"
        )
    model = MultiScaleWakeNet(**pack["model_kwargs"]).to(device)
    model.load_state_dict(pack["state_dict"])
    return model


def _auto_variant_name(bundle) -> str:
    scale_set = set(bundle.scale_names)
    if {"distD1.0_full", "distD2.0_full", "distD4.0_full"}.issubset(scale_set):
        return "distD_multi_4ch"
    if {"dist0.5_full", "dist1.0_full", "dist2.0_full"}.issubset(scale_set):
        return "dist_multi_4ch"
    raise RuntimeError(f"Cannot infer wake variant from scales={bundle.scale_names}")


def _instantiate_cfd_only_model(
    *,
    x_all: np.ndarray,
    n_shapes: int,
    n_re_classes: int,
    fusion_hidden: int,
    dropout: float,
    device: torch.device,
) -> torch.nn.Module:
    return MultiScaleWakeNet(
        n_scales=int(x_all.shape[1]),
        in_channels=int(x_all.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        fusion_hidden=int(fusion_hidden),
        dropout=float(dropout),
    ).to(device)


def _instantiate_smallcnn_model(
    *,
    x_all: np.ndarray,
    n_shapes: int,
    n_re_classes: int,
    feature_dim: int,
    fusion_hidden: int,
    dropout: float,
    encoder_norm: str,
    device: torch.device,
) -> torch.nn.Module:
    return MultiScaleJEPAModel(
        n_scales=int(x_all.shape[1]),
        in_channels=int(x_all.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        feature_dim=int(feature_dim),
        fusion_hidden=int(fusion_hidden),
        dropout=float(dropout),
        encoder_norm=str(encoder_norm),
        pretrained_encoder=None,
    ).to(device)


def _cfd_only_cfg(args: argparse.Namespace) -> dict[str, object]:
    return {
        "vision": {
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": 1e-4,
                "fusion_hidden": int(args.fusion_hidden),
                "dropout": float(args.dropout),
                "loss_weights": {
                    "shape": float(args.shape_weight),
                    "params": float(args.params_weight),
                    "re": float(args.re_weight),
                },
                "augmentation": {
                    "enabled": True,
                    "random_noise_std": float(args.noise_std),
                    "random_vertical_flip": 0.0,
                },
                "early_stopping_patience": 0,
            },
            "jepa": {
                "feature_dim": int(args.fusion_hidden),
                "fusion_hidden": int(args.fusion_hidden),
                "dropout": float(args.dropout),
                "encoder_norm": str(args.encoder_norm),
                "batch_size": int(args.batch_size),
                "pretrain_epochs": int(args.pretrain_epochs),
                "fine_tune_epochs": int(args.epochs),
                "lr": float(args.pretrain_lr),
                "fine_tune_lr": float(args.lr),
                "mask_ratio": float(args.mask_ratio),
                "loss_weights": {
                    "shape": float(args.shape_weight),
                    "params": float(args.params_weight),
                    "re": float(args.re_weight),
                },
                "early_stopping_patience": 0,
            },
        }
    }


def _labels(bundle, shape_labels: list[str], re_values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    shape_to_idx = {shape: idx for idx, shape in enumerate(shape_labels)}
    re_to_idx = {int(re_value): idx for idx, re_value in enumerate(re_values)}
    missing_shapes = sorted(set(bundle.shapes.tolist()) - set(shape_to_idx))
    missing_re = sorted(set(int(value) for value in bundle.re_values.tolist()) - set(re_to_idx))
    if missing_shapes:
        raise ValueError(f"CFD shapes missing from synthetic checkpoint labels: {missing_shapes}")
    if missing_re:
        raise ValueError(f"CFD Re values missing from synthetic checkpoint labels: {missing_re}")
    shape_idx = np.asarray([shape_to_idx[value] for value in bundle.shapes], dtype=np.int64)
    re_idx = np.asarray([re_to_idx[int(value)] for value in bundle.re_values], dtype=np.int64)
    return shape_idx, re_idx


def _params(bundle) -> np.ndarray:
    return np.stack([bundle.dy, bundle.eps], axis=1).astype(np.float32)


def _split_indices(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "split" in index_df:
        train_idx = index_df.index[index_df["split"].astype(str) == "train"].to_numpy(dtype=int)
        test_idx = index_df.index[index_df["split"].astype(str) == "test"].to_numpy(dtype=int)
    else:
        train_idx = index_df.index[np.abs(index_df["dy"].astype(float)) > 1.0e-12].to_numpy(
            dtype=int
        )
        test_idx = index_df.index[np.abs(index_df["dy"].astype(float)) <= 1.0e-12].to_numpy(
            dtype=int
        )
    return train_idx, test_idx


def _metrics(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    y_shape: np.ndarray,
    y_re: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if indices.size == 0:
        return {
            "n": 0.0,
            "shape_accuracy": float("nan"),
            "shape_macro_f1": float("nan"),
            "re_accuracy": float("nan"),
        }
    pred = predict_wake_model(model, x[indices], batch_size=batch_size, device=device)
    shape_pred = pred["shape_logits"].argmax(axis=1)
    re_pred = pred["re_logits"].argmax(axis=1)
    return {
        "n": float(indices.size),
        "shape_accuracy": float(accuracy_score(y_shape[indices], shape_pred)),
        "shape_macro_f1": float(f1_score(y_shape[indices], shape_pred, average="macro")),
        "re_accuracy": float(accuracy_score(y_re[indices], re_pred)),
    }


def _freeze_encoder(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = False


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _recalibrate_batch_norm(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    bn_layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers or x.shape[0] == 0:
        return
    previous_momentum = [module.momentum for module in bn_layers]
    for module in bn_layers:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch_x,) in loader:
            model(batch_x.to(device))
    for module, momentum in zip(bn_layers, previous_momentum):
        module.momentum = momentum
    model.eval()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    synthetic_model = (
        Path(args.synthetic_model).expanduser().resolve() if args.synthetic_model else None
    )
    cfd_run_dir = Path(args.cfd_run_dir).expanduser().resolve()
    output_run_dir = Path(args.output_run_dir).expanduser().resolve()
    reports_dir = output_run_dir / "reports"
    models_dir = output_run_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    wake_fields_dir = cfd_run_dir / "data" / "wake_fields"
    index_df = (
        pd.read_csv(wake_fields_dir / "index.csv").sort_values("case_id").reset_index(drop=True)
    )
    bundle = load_wake_bundle(wake_fields_dir)
    if synthetic_model is not None:
        pack = torch.load(synthetic_model, map_location="cpu", weights_only=False)
        variant_name = str(pack["variant_name"])
    else:
        variant_name = _auto_variant_name(bundle) if args.variant == "auto" else str(args.variant)
        pack = {}
    if variant_name not in VARIANTS:
        raise RuntimeError(f"Unknown checkpoint variant: {variant_name}")
    spec = VARIANTS[variant_name]
    x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])
    shape_labels = (
        [str(value) for value in pack["shape_labels"]]
        if pack
        else sorted(str(value) for value in np.unique(bundle.shapes))
    )
    re_values = (
        [int(value) for value in pack["re_values"]]
        if pack
        else sorted(int(value) for value in np.unique(bundle.re_values))
    )
    shape_idx, re_idx = _labels(bundle, shape_labels=shape_labels, re_values=re_values)
    params = _params(bundle)
    train_idx, test_idx = _split_indices(index_df)
    if args.train_all:
        train_idx = np.arange(index_df.shape[0], dtype=int)
        test_idx = np.asarray([], dtype=int)

    if pack:
        model = _instantiate_model(pack, device=device)
        if not args.unfreeze_encoder:
            _freeze_encoder(model)
        phase_prefix = "synthetic_checkpoint"
    else:
        if args.backbone == "resnet18":
            model = _instantiate_cfd_only_model(
                x_all=x_all,
                n_shapes=len(shape_labels),
                n_re_classes=len(re_values),
                fusion_hidden=args.fusion_hidden,
                dropout=args.dropout,
                device=device,
            )
        else:
            model = _instantiate_smallcnn_model(
                x_all=x_all,
                n_shapes=len(shape_labels),
                n_re_classes=len(re_values),
                feature_dim=args.fusion_hidden,
                fusion_hidden=args.fusion_hidden,
                dropout=args.dropout,
                encoder_norm=args.encoder_norm,
                device=device,
            )
        phase_prefix = "cfd_only_initial"

    metric_rows: list[dict[str, object]] = []
    metric_rows.append(
        {
            "phase": "synthetic_checkpoint",
            "split": "test",
            **_metrics(
                model=model,
                x=x_all,
                y_shape=shape_idx,
                y_re=re_idx,
                indices=test_idx,
                batch_size=args.batch_size,
                device=device,
            ),
        }
    )

    metric_rows[-1]["phase"] = phase_prefix

    if train_idx.size >= len(shape_labels):
        if pack or args.backbone == "resnet18":
            train_loader = tensor_loader(
                x_all[train_idx],
                shape_idx[train_idx],
                params[train_idx],
                re_idx[train_idx],
                batch_size=args.batch_size,
                shuffle=True,
            )
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.lr,
                weight_decay=1e-4,
            )
            history = _run_supervised_epochs(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                optimizer=optimizer,
                scheduler=None,
                epochs=args.epochs,
                device=device,
                loss_weights={
                    "shape": float(args.shape_weight),
                    "params": float(args.params_weight),
                    "re": float(args.re_weight),
                },
                aug_cfg={"enabled": True, "random_noise_std": float(args.noise_std)},
                patience=0,
            )
        else:
            model, history = train_wake_backbone(
                backbone=args.backbone,
                x_train=x_all[train_idx],
                shape_train_idx=shape_idx[train_idx],
                params_train=params[train_idx],
                re_train_idx=re_idx[train_idx],
                x_val=x_all[:0],
                shape_val_idx=shape_idx[:0],
                params_val=params[:0],
                re_val_idx=re_idx[:0],
                cfg=_cfd_only_cfg(args),
                seed=int(args.seed),
                n_shapes=len(shape_labels),
                n_re_classes=len(re_values),
                device=device,
            )
        _recalibrate_batch_norm(
            model,
            x_all[train_idx],
            batch_size=args.batch_size,
            device=device,
        )
        pd.DataFrame(history).to_csv(reports_dir / "cfd_finetune_history.csv", index=False)
        metric_rows.append(
            {
                "phase": "cfd_finetuned",
                "split": "train",
                **_metrics(
                    model=model,
                    x=x_all,
                    y_shape=shape_idx,
                    y_re=re_idx,
                    indices=train_idx,
                    batch_size=args.batch_size,
                    device=device,
                ),
            }
        )
        metric_rows.append(
            {
                "phase": "cfd_finetuned",
                "split": "test",
                **_metrics(
                    model=model,
                    x=x_all,
                    y_shape=shape_idx,
                    y_re=re_idx,
                    indices=test_idx,
                    batch_size=args.batch_size,
                    device=device,
                ),
            }
        )
    else:
        (reports_dir / "cfd_finetune_skipped.txt").write_text(
            f"Skipped fine-tune: train cases={train_idx.size}, shapes={len(shape_labels)}\n",
            encoding="utf-8",
        )

    output_pack = dict(pack)
    if not output_pack:
        output_pack.update(
            {
                "model_type": "resnet18",
                "variant_name": variant_name,
                "model_kwargs": {
                    "n_scales": int(x_all.shape[1]),
                    "in_channels": int(x_all.shape[2]),
                    "n_shapes": len(shape_labels),
                    "n_re_classes": len(re_values),
                    "fusion_hidden": int(args.fusion_hidden),
                    "dropout": float(args.dropout),
                },
                "shape_labels": shape_labels,
                "re_values": re_values,
                "cfg": {},
                "seed": 0,
            }
        )
        if args.backbone in {"smallcnn", "jepa"}:
            output_pack["model_type"] = args.backbone
            output_pack["model_kwargs"] = {
                "n_scales": int(x_all.shape[1]),
                "in_channels": int(x_all.shape[2]),
                "n_shapes": len(shape_labels),
                "n_re_classes": len(re_values),
                "feature_dim": int(args.fusion_hidden),
                "fusion_hidden": int(args.fusion_hidden),
                "dropout": float(args.dropout),
                "encoder_norm": str(args.encoder_norm),
            }
    output_pack["state_dict"] = model.state_dict()
    output_pack["fine_tune"] = {
        "source_checkpoint": str(synthetic_model) if synthetic_model else None,
        "cfd_run_dir": str(cfd_run_dir),
        "train_case_ids": bundle.case_ids[train_idx].tolist(),
        "test_case_ids": bundle.case_ids[test_idx].tolist(),
        "encoder_frozen": not args.unfreeze_encoder,
        "train_all": bool(args.train_all),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "loss_weights": {
            "shape": float(args.shape_weight),
            "params": float(args.params_weight),
            "re": float(args.re_weight),
        },
        "noise_std": float(args.noise_std),
    }
    torch.save(output_pack, models_dir / "wake_field_main_cfd_finetuned.pt")
    _write_metrics(reports_dir / "cfd_finetune_metrics.csv", metric_rows)
    if synthetic_model is not None:
        shutil.copy2(synthetic_model, models_dir / "source_synthetic_checkpoint.pt")
    (reports_dir / "cfd_finetune_summary.json").write_text(
        json.dumps(
            {
                "synthetic_model": str(synthetic_model),
                "cfd_run_dir": str(cfd_run_dir),
                "output_model": str(models_dir / "wake_field_main_cfd_finetuned.pt"),
                "metrics_csv": str(reports_dir / "cfd_finetune_metrics.csv"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CFD fine-tune complete: {reports_dir / 'cfd_finetune_metrics.csv'}")


if __name__ == "__main__":
    main()
