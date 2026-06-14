from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.wake_common import VARIANTS
from ml.wake_training import (
    _run_supervised_epochs,
    predict_wake_model,
    set_seed,
    tensor_loader,
)
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleJEPAModel, MultiScaleWakeNet, select_device

DEFAULT_PIV_RUN = Path("/home/chenyihao/fluid_runs/piv_blueluna_validation_stride5")
DEFAULT_CHECKPOINT = Path(
    "/home/chenyihao/fluid_runs/cfd_final_stable175_tau6_all_jepa_gn/"
    "models/wake_field_main_cfd_finetuned.pt"
)
DEFAULT_OUTPUT = Path("/home/chenyihao/fluid_runs/piv_jepa_cfd_init_seq12_test3_stride5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a CFD/synthetic wake checkpoint on real PIV wake tensors"
    )
    parser.add_argument("--piv-run-dir", type=Path, default=DEFAULT_PIV_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-run-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-sequences", default="1,2")
    parser.add_argument("--test-sequences", default="3")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--variant", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--params-weight", type=float, default=0.0)
    parser.add_argument("--re-weight", type=float, default=0.0)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the JEPA/CNN encoder and only fine-tune fusion plus heads.",
    )
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


def _instantiate_model(
    pack: dict, *, device: torch.device, n_re_classes: int
) -> tuple[torch.nn.Module, dict[str, object]]:
    model_type = str(pack.get("model_type", "resnet18"))
    model_kwargs = dict(pack["model_kwargs"])
    model_kwargs["n_re_classes"] = int(n_re_classes)
    if model_type == "jepa":
        model = MultiScaleJEPAModel(**model_kwargs).to(device)
    elif model_type == "resnet18":
        model = MultiScaleWakeNet(**model_kwargs).to(device)
    else:
        raise RuntimeError(f"Unsupported checkpoint model_type={model_type}")
    model_state = model.state_dict()
    checkpoint_state = pack["state_dict"]
    compatible_state = {}
    skipped = []
    for name, tensor in checkpoint_state.items():
        if name not in model_state:
            skipped.append(name)
            continue
        if tuple(model_state[name].shape) != tuple(tensor.shape):
            skipped.append(name)
            continue
        compatible_state[name] = tensor
    load_result = model.load_state_dict(compatible_state, strict=False)
    return model, {
        "loaded_keys": len(compatible_state),
        "skipped_keys": skipped,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _variant_name(bundle, pack: dict, requested: str) -> str:
    if requested != "auto":
        return str(requested)
    checkpoint_variant = pack.get("variant_name")
    if checkpoint_variant:
        return str(checkpoint_variant)
    scale_set = set(bundle.scale_names)
    if {"distD1.0_full", "distD2.0_full", "distD4.0_full"}.issubset(scale_set):
        return "distD_multi_4ch"
    if {"dist0.5_full", "dist1.0_full", "dist2.0_full"}.issubset(scale_set):
        return "dist_multi_4ch"
    raise RuntimeError(f"Cannot infer wake variant from scales={bundle.scale_names}")


def _shape_indices(shapes: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    missing = sorted(set(shapes.tolist()) - set(label_to_idx))
    if missing:
        raise ValueError(f"PIV shapes missing from checkpoint labels: {missing}")
    return np.asarray([label_to_idx[str(shape)] for shape in shapes], dtype=np.int64)


def _re_indices(re_values: np.ndarray, labels: list[int]) -> np.ndarray:
    re_to_idx = {int(re_value): idx for idx, re_value in enumerate(labels)}
    return np.asarray([re_to_idx[int(value)] for value in re_values], dtype=np.int64)


def _params(bundle) -> np.ndarray:
    return np.stack([bundle.dy, bundle.eps], axis=1).astype(np.float32)


def _split_indices(
    index_df: pd.DataFrame, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(index_df), dtype=bool)
    speed_levels = _parse_int_set(args.speed_levels)
    if speed_levels:
        mask &= index_df["speed_level"].astype(int).isin(speed_levels).to_numpy()
    if args.train_all:
        return np.flatnonzero(mask), np.asarray([], dtype=np.int64)
    train_sequences = _parse_int_set(args.train_sequences)
    test_sequences = _parse_int_set(args.test_sequences)
    train_mask = mask & index_df["sequence"].astype(int).isin(train_sequences).to_numpy()
    test_mask = mask & index_df["sequence"].astype(int).isin(test_sequences).to_numpy()
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _frame_predictions(
    index_df: pd.DataFrame,
    logits: np.ndarray,
    probabilities: np.ndarray,
    shape_labels: list[str],
) -> pd.DataFrame:
    pred_idx = logits.argmax(axis=1)
    output = index_df.copy()
    output["pred_shape_idx"] = pred_idx
    output["pred_shape"] = [shape_labels[int(idx)] for idx in pred_idx]
    output["pred_confidence"] = probabilities.max(axis=1)
    output["correct"] = output["shape"].astype(str) == output["pred_shape"].astype(str)
    for idx, label in enumerate(shape_labels):
        output[f"prob_{label}"] = probabilities[:, idx]
    return output


def _sequence_predictions(frame_df: pd.DataFrame, shape_labels: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["source_case_id", "shape", "speed_level", "sequence"]
    for key, group in frame_df.groupby(group_cols, dropna=False):
        source_case_id, shape, speed_level, sequence = key
        prob_means = {label: float(group[f"prob_{label}"].mean()) for label in shape_labels}
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
                **{f"mean_prob_{label}": prob_means[label] for label in shape_labels},
            }
        )
    return pd.DataFrame(rows).sort_values(["shape", "speed_level", "sequence"])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _evaluate_split(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    index_df: pd.DataFrame,
    indices: np.ndarray,
    shape_labels: list[str],
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
        index_df.iloc[indices].reset_index(drop=True),
        pred["shape_logits"],
        probabilities,
        shape_labels,
    )
    frame_df.to_csv(output_dir / f"{split_name}_frame_predictions.csv", index=False)
    sequence_df = _sequence_predictions(frame_df, shape_labels)
    sequence_df.to_csv(output_dir / f"{split_name}_sequence_predictions.csv", index=False)

    label_to_idx = {label: idx for idx, label in enumerate(shape_labels)}
    y_frame = np.asarray([label_to_idx[str(v)] for v in frame_df["shape"]], dtype=np.int64)
    y_frame_pred = frame_df["pred_shape_idx"].to_numpy(dtype=np.int64)
    y_seq = np.asarray([label_to_idx[str(v)] for v in sequence_df["shape"]], dtype=np.int64)
    y_seq_pred = np.asarray(
        [label_to_idx[str(v)] for v in sequence_df["pred_shape_mean_prob"]],
        dtype=np.int64,
    )
    frame_metrics = _metrics(y_frame, y_frame_pred)
    sequence_metrics = _metrics(y_seq, y_seq_pred)
    cm = confusion_matrix(y_seq, y_seq_pred, labels=np.arange(len(shape_labels)))
    pd.DataFrame(cm, index=shape_labels, columns=shape_labels).to_csv(
        output_dir / f"{split_name}_sequence_confusion_matrix.csv"
    )
    return {"split": split_name, "frame": frame_metrics, "sequence": sequence_metrics}


def _freeze_encoder(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = False


def fine_tune(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    piv_run_dir = args.piv_run_dir.expanduser().resolve()
    wake_dir = piv_run_dir / "data" / "wake_fields"
    output_run_dir = args.output_run_dir.expanduser().resolve()
    report_dir = output_run_dir / "reports"
    model_dir = output_run_dir / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(wake_dir / "index.csv").sort_values("case_id").reset_index(drop=True)
    bundle = load_wake_bundle(wake_dir)
    pack = torch.load(
        args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    variant_name = _variant_name(bundle, pack, str(args.variant))
    variant = VARIANTS[variant_name]
    x = variant_tensor(bundle, scales=variant["scales"], channels=variant["channels"])
    shape_labels = [str(label) for label in pack["shape_labels"]]
    re_values = sorted(int(value) for value in np.unique(bundle.re_values))
    y_shape = _shape_indices(bundle.shapes, shape_labels)
    y_re = _re_indices(bundle.re_values, re_values)
    params = _params(bundle)
    train_idx, test_idx = _split_indices(index_df, args)
    if train_idx.size == 0:
        raise RuntimeError("No PIV training rows selected")

    device = select_device()
    model, load_report = _instantiate_model(pack, device=device, n_re_classes=len(re_values))
    if args.freeze_encoder:
        _freeze_encoder(model)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    train_loader = tensor_loader(
        x[train_idx],
        y_shape[train_idx],
        params[train_idx],
        y_re[train_idx],
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=None,
        epochs=int(args.epochs),
        device=device,
        loss_weights={
            "shape": float(args.shape_weight),
            "params": float(args.params_weight),
            "re": float(args.re_weight),
        },
        aug_cfg={
            "enabled": True,
            "random_noise_std": float(args.noise_std),
            "random_vertical_flip": 0.0,
        },
        patience=0,
    )
    pd.DataFrame(history).to_csv(report_dir / "piv_checkpoint_finetune_history.csv", index=False)

    metrics = [
        _evaluate_split(
            model=model,
            x=x,
            index_df=index_df,
            indices=train_idx,
            shape_labels=shape_labels,
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
                shape_labels=shape_labels,
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
    metrics_df.to_csv(report_dir / "piv_checkpoint_finetune_metrics.csv", index=False)

    output_pack = dict(pack)
    output_pack["model_kwargs"] = dict(pack["model_kwargs"])
    output_pack["model_kwargs"]["n_re_classes"] = len(re_values)
    output_pack["re_values"] = re_values
    output_pack["state_dict"] = model.state_dict()
    output_pack["fine_tune"] = {
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "piv_run_dir": str(piv_run_dir),
        "variant": variant_name,
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "train_sequences": str(args.train_sequences),
        "test_sequences": str(args.test_sequences),
        "train_all": bool(args.train_all),
        "speed_levels": args.speed_levels or "all",
        "freeze_encoder": bool(args.freeze_encoder),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "loss_weights": {
            "shape": float(args.shape_weight),
            "params": float(args.params_weight),
            "re": float(args.re_weight),
        },
        "noise_std": float(args.noise_std),
        "checkpoint_load": load_report,
    }
    model_path = model_dir / "wake_field_main_piv_from_checkpoint.pt"
    torch.save(output_pack, model_path)
    shutil.copy2(args.checkpoint.expanduser().resolve(), model_dir / "source_checkpoint.pt")
    summary = {
        "checkpoint": str(args.checkpoint),
        "piv_run_dir": str(piv_run_dir),
        "output_run_dir": str(output_run_dir),
        "model_path": str(model_path),
        "variant": variant_name,
        "device": str(device),
        "shape_labels": shape_labels,
        "re_values": re_values,
        "checkpoint_load": load_report,
        "metrics": metrics_rows,
        "notes": [
            "Fine-tuning uses only the selected train sequences.",
            "The test sequence is evaluated after training and is not used for early stopping.",
            "Re and geometry losses are disabled by default for real PIV shape fine-tuning.",
        ],
    }
    (report_dir / "piv_checkpoint_finetune_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))
    print(f"model={model_path}")
    print(f"reports={report_dir}")


def main() -> None:
    fine_tune(parse_args())


if __name__ == "__main__":
    main()
