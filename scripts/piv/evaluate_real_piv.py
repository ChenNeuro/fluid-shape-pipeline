from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.wake_common import VARIANTS
from ml.wake_training import predict_wake_model
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleJEPAModel, MultiScaleWakeNet, select_device

DEFAULT_MODEL = Path(
    "/home/chenyihao/fluid_runs/cfd_final_stable175_tau6_all_jepa_gn/"
    "models/wake_field_main_cfd_finetuned.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained wake model on real PIV data")
    parser.add_argument(
        "--piv-run-dir",
        type=Path,
        default=Path("/home/chenyihao/fluid_runs/piv_blueluna_validation"),
        help="Run directory containing data/wake_fields/index.csv",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/chenyihao/fluid_runs/piv_blueluna_validation/reports"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variant", default="auto")
    return parser.parse_args()


def _instantiate_model(pack: dict, device: torch.device) -> torch.nn.Module:
    model_type = str(pack.get("model_type", "resnet18"))
    if model_type == "jepa":
        model = MultiScaleJEPAModel(**pack["model_kwargs"]).to(device)
    elif model_type == "resnet18":
        model = MultiScaleWakeNet(**pack["model_kwargs"]).to(device)
    else:
        raise RuntimeError(f"Unsupported checkpoint model_type={model_type}")
    model.load_state_dict(pack["state_dict"])
    return model


def _variant_name(bundle, pack: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    checkpoint_variant = pack.get("variant_name")
    if checkpoint_variant:
        return str(checkpoint_variant)
    scale_set = set(bundle.scale_names)
    if {"distD1.0_full", "distD2.0_full", "distD4.0_full"}.issubset(scale_set):
        return "distD_multi_4ch"
    if {"dist0.5_full", "dist1.0_full", "dist2.0_full"}.issubset(scale_set):
        return "dist_multi_4ch"
    raise RuntimeError(f"Cannot infer variant from scales={bundle.scale_names}")


def _shape_indices(shapes: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    missing = sorted(set(shapes.tolist()) - set(label_to_idx))
    if missing:
        raise ValueError(f"PIV shapes missing from checkpoint labels: {missing}")
    return np.asarray([label_to_idx[str(shape)] for shape in shapes], dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _frame_predictions(
    *,
    index_df: pd.DataFrame,
    logits: np.ndarray,
    probabilities: np.ndarray,
    shape_labels: list[str],
) -> pd.DataFrame:
    pred_idx = logits.argmax(axis=1)
    output = index_df.copy()
    output["pred_shape"] = [shape_labels[int(idx)] for idx in pred_idx]
    output["pred_shape_idx"] = pred_idx
    output["pred_confidence"] = probabilities.max(axis=1)
    for idx, label in enumerate(shape_labels):
        output[f"prob_{label}"] = probabilities[:, idx]
    output["correct"] = output["shape"].astype(str) == output["pred_shape"].astype(str)
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


def evaluate(args: argparse.Namespace) -> None:
    wake_dir = args.piv_run_dir.expanduser().resolve() / "data" / "wake_fields"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = pd.read_csv(wake_dir / "index.csv").sort_values("case_id").reset_index(drop=True)
    bundle = load_wake_bundle(wake_dir)
    pack = torch.load(args.model.expanduser().resolve(), map_location="cpu")
    variant_name = _variant_name(bundle, pack, args.variant)
    variant = VARIANTS[variant_name]
    x = variant_tensor(bundle, scales=variant["scales"], channels=variant["channels"])
    shape_labels = [str(label) for label in pack["shape_labels"]]
    y_true = _shape_indices(bundle.shapes, shape_labels)

    device = select_device()
    model = _instantiate_model(pack, device)
    predictions = predict_wake_model(model, x, batch_size=int(args.batch_size), device=device)
    probabilities = _softmax(predictions["shape_logits"])
    frame_df = _frame_predictions(
        index_df=index_df,
        logits=predictions["shape_logits"],
        probabilities=probabilities,
        shape_labels=shape_labels,
    )
    frame_df.to_csv(output_dir / "piv_frame_predictions.csv", index=False)

    seq_df = _sequence_predictions(frame_df, shape_labels)
    seq_df.to_csv(output_dir / "piv_sequence_predictions.csv", index=False)

    frame_pred_idx = frame_df["pred_shape_idx"].to_numpy(dtype=np.int64)
    seq_true = _shape_indices(seq_df["shape"].to_numpy(dtype=str), shape_labels)
    seq_pred_mean = _shape_indices(seq_df["pred_shape_mean_prob"].to_numpy(dtype=str), shape_labels)
    seq_pred_vote = _shape_indices(
        seq_df["pred_shape_majority_vote"].to_numpy(dtype=str), shape_labels
    )
    metrics_rows = [
        {"level": "frame", **_metrics(y_true, frame_pred_idx)},
        {"level": "sequence_mean_prob", **_metrics(seq_true, seq_pred_mean)},
        {"level": "sequence_majority_vote", **_metrics(seq_true, seq_pred_vote)},
    ]
    pd.DataFrame(metrics_rows).to_csv(output_dir / "piv_metrics.csv", index=False)

    per_speed = []
    for speed_level, group in seq_df.groupby("speed_level"):
        true_idx = _shape_indices(group["shape"].to_numpy(dtype=str), shape_labels)
        pred_idx = _shape_indices(group["pred_shape_mean_prob"].to_numpy(dtype=str), shape_labels)
        per_speed.append({"speed_level": speed_level, **_metrics(true_idx, pred_idx)})
    pd.DataFrame(per_speed).to_csv(output_dir / "piv_metrics_by_speed_level.csv", index=False)

    cm = confusion_matrix(seq_true, seq_pred_mean, labels=np.arange(len(shape_labels)))
    pd.DataFrame(cm, index=shape_labels, columns=shape_labels).to_csv(
        output_dir / "piv_sequence_confusion_matrix.csv"
    )
    summary = {
        "piv_run_dir": str(args.piv_run_dir),
        "model": str(args.model),
        "variant": variant_name,
        "device": str(device),
        "shape_labels": shape_labels,
        "metrics": metrics_rows,
        "notes": [
            "Sequence metrics are more trustworthy than frame metrics because "
            "frames are correlated.",
            "Real PIV crop geometry is not identical to CFD distD crop geometry.",
        ],
    }
    (output_dir / "piv_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(f"reports={output_dir}")


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
