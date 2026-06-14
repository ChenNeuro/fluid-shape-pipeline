from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.wake_common import VARIANTS
from ml.wake_training import predict_wake_model, save_model_pack, train_wake_backbone
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import select_device

DEFAULT_PIV_RUN = Path("/home/chenyihao/fluid_runs/piv_blueluna_validation_stride5")
DEFAULT_OUTPUT = Path("/home/chenyihao/fluid_runs/piv_jepa_seq12_test3_stride5")
SHAPE_LABELS = ["airfoil", "bar", "circle", "diamond", "triangle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a JEPA wake classifier on real PIV wake-field tensors"
    )
    parser.add_argument("--piv-run-dir", type=Path, default=DEFAULT_PIV_RUN)
    parser.add_argument("--output-run-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-sequences", default="1,2")
    parser.add_argument("--test-sequences", default="3")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Use all PIV rows for final training; still writes no internal test metrics.",
    )
    parser.add_argument("--variant", default="distD_multi_4ch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=192)
    parser.add_argument("--fusion-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--encoder-norm", choices=["batch", "group", "identity"], default="group")
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--params-weight", type=float, default=0.0)
    parser.add_argument("--re-weight", type=float, default=0.0)
    parser.add_argument(
        "--speed-levels",
        default="",
        help="Optional comma list such as 5,10,15 to filter speed levels.",
    )
    return parser.parse_args()


def _parse_int_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _cfg(args: argparse.Namespace) -> dict:
    return {
        "vision": {
            "training": {
                "batch_size": int(args.batch_size),
                "scheduler": {"type": "cosine", "T_max": int(args.epochs)},
                "augmentation": {
                    "enabled": True,
                    "random_noise_std": float(args.noise_std),
                    "random_vertical_flip": 0.0,
                },
                "loss_weights": {
                    "shape": 1.0,
                    "params": float(args.params_weight),
                    "re": float(args.re_weight),
                },
            },
            "jepa": {
                "batch_size": int(args.batch_size),
                "pretrain_epochs": int(args.pretrain_epochs),
                "fine_tune_epochs": int(args.epochs),
                "feature_dim": int(args.feature_dim),
                "fusion_hidden": int(args.fusion_hidden),
                "dropout": float(args.dropout),
                "encoder_norm": str(args.encoder_norm),
                "lr": float(args.pretrain_lr),
                "fine_tune_lr": float(args.lr),
                "mask_ratio": float(args.mask_ratio),
                "loss_weights": {
                    "shape": 1.0,
                    "params": float(args.params_weight),
                    "re": float(args.re_weight),
                },
                "early_stopping_patience": 0,
            },
        }
    }


def _shape_indices(shapes: np.ndarray) -> np.ndarray:
    shape_to_idx = {shape: idx for idx, shape in enumerate(SHAPE_LABELS)}
    missing = sorted(set(shapes.tolist()) - set(shape_to_idx))
    if missing:
        raise ValueError(f"PIV shapes missing from labels: {missing}")
    return np.asarray([shape_to_idx[str(shape)] for shape in shapes], dtype=np.int64)


def _re_indices(re_values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    unique_re = sorted(int(value) for value in np.unique(re_values))
    re_to_idx = {value: idx for idx, value in enumerate(unique_re)}
    return np.asarray([re_to_idx[int(value)] for value in re_values], dtype=np.int64), unique_re


def _params(bundle) -> np.ndarray:
    return np.stack([bundle.dy, bundle.eps], axis=1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _frame_predictions(
    index_df: pd.DataFrame, logits: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    pred_idx = logits.argmax(axis=1)
    output = index_df.copy()
    output["pred_shape_idx"] = pred_idx
    output["pred_shape"] = [SHAPE_LABELS[int(idx)] for idx in pred_idx]
    output["pred_confidence"] = probabilities.max(axis=1)
    output["correct"] = output["shape"].astype(str) == output["pred_shape"].astype(str)
    for idx, label in enumerate(SHAPE_LABELS):
        output[f"prob_{label}"] = probabilities[:, idx]
    return output


def _sequence_predictions(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_case_id", "shape", "speed_level", "sequence"]
    for key, group in frame_df.groupby(group_cols, dropna=False):
        source_case_id, shape, speed_level, sequence = key
        prob_means = {label: float(group[f"prob_{label}"].mean()) for label in SHAPE_LABELS}
        pred_shape = max(prob_means, key=prob_means.get)
        vote_shape = str(group["pred_shape"].mode().iloc[0])
        rows.append(
            {
                "source_case_id": source_case_id,
                "shape": shape,
                "speed_level": speed_level,
                "sequence": sequence,
                "n_frames": int(len(group)),
                "pred_shape_mean_prob": pred_shape,
                "pred_shape_majority_vote": vote_shape,
                "mean_confidence": float(group["pred_confidence"].mean()),
                "correct_mean_prob": str(shape) == pred_shape,
                "correct_majority_vote": str(shape) == vote_shape,
                **{f"mean_prob_{label}": prob_means[label] for label in SHAPE_LABELS},
            }
        )
    return pd.DataFrame(rows).sort_values(["shape", "speed_level", "sequence"])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _shape_idx_from_labels(labels: np.ndarray) -> np.ndarray:
    shape_to_idx = {shape: idx for idx, shape in enumerate(SHAPE_LABELS)}
    return np.asarray([shape_to_idx[str(label)] for label in labels], dtype=np.int64)


def _evaluate_split(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    index_df: pd.DataFrame,
    indices: np.ndarray,
    output_dir: Path,
    split_name: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if indices.size == 0:
        return {"split": split_name, "frame": {"n": 0.0}, "sequence": {"n": 0.0}}
    pred = predict_wake_model(model, x[indices], batch_size=batch_size, device=device)
    probabilities = _softmax(pred["shape_logits"])
    frame_df = _frame_predictions(
        index_df.iloc[indices].reset_index(drop=True), pred["shape_logits"], probabilities
    )
    frame_df.to_csv(output_dir / f"{split_name}_frame_predictions.csv", index=False)
    sequence_df = _sequence_predictions(frame_df)
    sequence_df.to_csv(output_dir / f"{split_name}_sequence_predictions.csv", index=False)

    y_frame = _shape_idx_from_labels(frame_df["shape"].to_numpy(dtype=str))
    y_frame_pred = frame_df["pred_shape_idx"].to_numpy(dtype=np.int64)
    y_seq = _shape_idx_from_labels(sequence_df["shape"].to_numpy(dtype=str))
    y_seq_pred = _shape_idx_from_labels(sequence_df["pred_shape_mean_prob"].to_numpy(dtype=str))
    frame_metrics = _metrics(y_frame, y_frame_pred)
    sequence_metrics = _metrics(y_seq, y_seq_pred)
    cm = confusion_matrix(y_seq, y_seq_pred, labels=np.arange(len(SHAPE_LABELS)))
    pd.DataFrame(cm, index=SHAPE_LABELS, columns=SHAPE_LABELS).to_csv(
        output_dir / f"{split_name}_sequence_confusion_matrix.csv"
    )
    return {"split": split_name, "frame": frame_metrics, "sequence": sequence_metrics}


def _split_indices(
    index_df: pd.DataFrame, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(index_df), dtype=bool)
    speed_levels = _parse_int_set(args.speed_levels)
    if speed_levels:
        mask &= index_df["speed_level"].astype(int).isin(speed_levels).to_numpy()
    train_sequences = _parse_int_set(args.train_sequences)
    test_sequences = _parse_int_set(args.test_sequences)
    if args.train_all:
        train_mask = mask
        test_mask = np.zeros(len(index_df), dtype=bool)
    else:
        train_mask = mask & index_df["sequence"].astype(int).isin(train_sequences).to_numpy()
        test_mask = mask & index_df["sequence"].astype(int).isin(test_sequences).to_numpy()
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def train(args: argparse.Namespace) -> None:
    wake_dir = args.piv_run_dir.expanduser().resolve() / "data" / "wake_fields"
    output_run_dir = args.output_run_dir.expanduser().resolve()
    report_dir = output_run_dir / "reports"
    model_dir = output_run_dir / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(wake_dir / "index.csv").sort_values("case_id").reset_index(drop=True)
    bundle = load_wake_bundle(wake_dir)
    variant = VARIANTS[str(args.variant)]
    x = variant_tensor(bundle, scales=variant["scales"], channels=variant["channels"])
    y_shape = _shape_indices(bundle.shapes)
    y_re, re_values = _re_indices(bundle.re_values)
    params = _params(bundle)
    train_idx, test_idx = _split_indices(index_df, args)
    if train_idx.size == 0:
        raise RuntimeError("No PIV training rows selected")

    device = select_device()
    cfg = _cfg(args)
    model, history = train_wake_backbone(
        backbone="jepa",
        x_train=x[train_idx],
        shape_train_idx=y_shape[train_idx],
        params_train=params[train_idx],
        re_train_idx=y_re[train_idx],
        x_val=x[test_idx],
        shape_val_idx=y_shape[test_idx],
        params_val=params[test_idx],
        re_val_idx=y_re[test_idx],
        cfg=cfg,
        seed=int(args.seed),
        n_shapes=len(SHAPE_LABELS),
        n_re_classes=len(re_values),
        device=device,
    )
    pd.DataFrame(history).to_csv(report_dir / "training_history.csv", index=False)
    model_path = model_dir / "wake_field_main_piv_jepa.pt"
    save_model_pack(
        output_path=model_path,
        model=model,
        model_type="jepa",
        variant_name=str(args.variant),
        x_shape=tuple(x.shape),
        shape_labels=SHAPE_LABELS,
        re_values=re_values,
        test_case_ids=index_df.iloc[test_idx]["case_id"].astype(str).tolist(),
        cfg=cfg,
        seed=int(args.seed),
    )
    metrics = [
        _evaluate_split(
            model=model,
            x=x,
            index_df=index_df,
            indices=train_idx,
            output_dir=report_dir,
            split_name="train",
            batch_size=int(args.batch_size),
            device=device,
        )
    ]
    if test_idx.size:
        metrics.append(
            _evaluate_split(
                model=model,
                x=x,
                index_df=index_df,
                indices=test_idx,
                output_dir=report_dir,
                split_name="test",
                batch_size=int(args.batch_size),
                device=device,
            )
        )
    metrics_rows = []
    for entry in metrics:
        for level in ["frame", "sequence"]:
            row = {"split": entry["split"], "level": level}
            row.update(entry[level])
            metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(report_dir / "piv_jepa_metrics.csv", index=False)
    summary = {
        "piv_run_dir": str(args.piv_run_dir),
        "output_run_dir": str(output_run_dir),
        "model_path": str(model_path),
        "variant": str(args.variant),
        "device": str(device),
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "train_all": bool(args.train_all),
        "speed_levels": args.speed_levels or "all",
        "shape_labels": SHAPE_LABELS,
        "re_values": re_values,
        "metrics": metrics_rows,
        "notes": [
            "The evaluation split is by independent sequence, not random frames.",
            "Re loss is disabled by default; shape classification is the primary task.",
        ],
    }
    (report_dir / "piv_jepa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))
    print(f"model={model_path}")
    print(f"reports={report_dir}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
