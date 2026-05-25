from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ml.wake_common import (
    VARIANTS,
    accuracy_f1,
    build_label_maps,
    clip_params,
    obstacle_iou_and_dice,
    render_targets,
)
from sim.config import load_config, repo_root
from sim.logging_utils import setup_logger
from vision.diff_renderer import DifferentiableShapeRenderer
from vision.wake_dataset import load_wake_bundle, variant_tensor
from vision.wake_model import MultiScaleWakeNet, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical inverse reconstruction from wake-field predictions"
    )
    parser.add_argument(
        "--config", default="configs/wake_field_450.yaml", help="Path to YAML config"
    )
    return parser.parse_args()


def _predict(
    model: torch.nn.Module, x: np.ndarray, *, batch_size: int, device: torch.device
) -> dict[str, np.ndarray]:
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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


def _plot_examples(
    *,
    case_ids: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    output_path: Path,
) -> None:
    n_show = int(min(4, targets.shape[0]))
    fig, axes = plt.subplots(n_show, 2, figsize=(8.8, 2.2 * n_show))
    if n_show == 1:
        axes = np.asarray([axes])

    for row_idx in range(n_show):
        ax_t, ax_p = axes[row_idx]
        ax_t.imshow(targets[row_idx], cmap="viridis", vmin=0.0, vmax=1.0)
        ax_t.set_title(f"{case_ids[row_idx]} target")
        ax_t.axis("off")
        ax_p.imshow(predictions[row_idx], cmap="viridis", vmin=0.0, vmax=1.0)
        ax_p.set_title(f"pred {shape_pred[row_idx]} (true {shape_true[row_idx]})")
        ax_p.axis("off")

    fig.suptitle("Wake-Field Canonical Inverse Reconstruction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(
    *,
    output_path: Path,
    variant_name: str,
    metrics: dict[str, float],
    sanity_metrics: dict[str, float],
) -> None:
    lines = [
        "# Wake-Field Reconstruction Summary",
        "",
        f"- Model variant: `{variant_name}`",
        f"- Shape accuracy: {metrics['accuracy']:.4f}",
        f"- Shape macro F1: {metrics['macro_f1']:.4f}",
        f"- dy MAE: {metrics['dy_mae']:.5f}",
        f"- eps MAE: {metrics['eps_mae']:.5f}",
        f"- Inverse obstacle IoU (binary): {metrics['inverse_iou']:.4f}",
        f"- Inverse obstacle Dice (binary): {metrics['inverse_dice']:.4f}",
        f"- Full-geometry Soft IoU (diff-renderer): {metrics['soft_iou']:.4f}",
        f"- Full-geometry Soft Dice (diff-renderer): {metrics['soft_dice']:.4f}",
        "",
        "## Sanity",
        f"- Ground-truth render self-check IoU: {sanity_metrics['inverse_iou']:.4f}",
        f"- Ground-truth render self-check Dice: {sanity_metrics['inverse_dice']:.4f}",
        f"- Full-geometry Soft IoU self-check: {sanity_metrics['soft_iou']:.4f}",
        f"- Full-geometry Soft Dice self-check: {sanity_metrics['soft_dice']:.4f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    logger = setup_logger("reconstruct_wake", root / "logs" / "reconstruct_wake.log")

    model_path = root / "models" / "wake_field_main.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing wake-field model pack: {model_path}. Run ml.train_wake first."
        )

    bundle = load_wake_bundle()
    label_maps = build_label_maps(bundle)
    device = select_device()
    batch_size = int(cfg.get("vision", {}).get("training", {}).get("batch_size", 16))
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pack = torch.load(model_path, map_location="cpu", weights_only=False)
    variant_name = str(pack["variant_name"])
    if variant_name not in VARIANTS:
        raise RuntimeError(f"Unknown wake-field variant stored in model pack: {variant_name}")
    spec = VARIANTS[variant_name]
    x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])

    case_to_idx = {case_id: idx for idx, case_id in enumerate(bundle.case_ids.tolist())}
    idx_test = np.asarray([case_to_idx[case_id] for case_id in pack["test_case_ids"]], dtype=int)

    model_type = str(pack.get("model_type", "resnet18"))
    if model_type == "mae_vit":
        from vision.mae_vit_model import MultiScaleViTWakeNet

        model = MultiScaleViTWakeNet(**pack["model_kwargs"]).to(device)
    else:
        model = MultiScaleWakeNet(**pack["model_kwargs"]).to(device)
    model.load_state_dict(pack["state_dict"])

    sim_cfg = cfg["simulation"]
    rec_cfg = cfg["reconstruction"]
    renderer = DifferentiableShapeRenderer(
        image_size=int(rec_cfg.get("image_height", 64)),
        l_total=float(sim_cfg.get("L_in", 5.0)) + float(sim_cfg.get("L_out", 5.0)),
        h=float(sim_cfg.get("H", 1.0)),
        d_ratio=float(sim_cfg.get("d_ratio", 0.2)),
        x0=float(sim_cfg.get("x0", 3.0)),
        y0=float(sim_cfg.get("y0", 0.5)),
        eps_max=float(sim_cfg.get("perturb", {}).get("eps_max", 0.06)),
    ).to(device)
    renderer.eval()

    pred = _predict(model, x_all[idx_test], batch_size=batch_size, device=device)
    shape_labels = list(pack["shape_labels"])
    shape_pred_idx = np.argmax(pred["shape_logits"], axis=1)
    shape_pred = np.asarray([shape_labels[int(idx)] for idx in shape_pred_idx], dtype=object)
    dy_pred, eps_pred = clip_params(pred["params_pred"][:, 0], pred["params_pred"][:, 1], cfg)

    dy_true = bundle.dy[idx_test]
    eps_true = bundle.eps[idx_test]
    shapes_true = bundle.shapes[idx_test]

    targets = render_targets(shapes=shapes_true, dy_values=dy_true, eps_values=eps_true, cfg=cfg)
    predictions = render_targets(shapes=shape_pred, dy_values=dy_pred, eps_values=eps_pred, cfg=cfg)
    sanity_predictions = render_targets(
        shapes=shapes_true, dy_values=dy_true, eps_values=eps_true, cfg=cfg
    )

    threshold = float(cfg["reconstruction"].get("obstacle_threshold", 0.8))
    inv_metrics = obstacle_iou_and_dice(targets, predictions, threshold=threshold)
    sanity_metrics = obstacle_iou_and_dice(targets, sanity_predictions, threshold=threshold)

    with torch.no_grad():
        shape_indices = torch.from_numpy(shape_pred_idx).long().to(device)
        dy_t = torch.from_numpy(dy_pred.astype(np.float32)).to(device)
        eps_t = torch.from_numpy(eps_pred.astype(np.float32)).to(device)
        pred_soft = renderer.render(shape_indices, dy_t, eps_t)

        true_shape_idx = np.asarray(
            [label_maps.shape_to_idx[s] for s in shapes_true], dtype=np.int64
        )
        true_shape_t = torch.from_numpy(true_shape_idx).long().to(device)
        true_dy_t = torch.from_numpy(dy_true.astype(np.float32)).to(device)
        true_eps_t = torch.from_numpy(eps_true.astype(np.float32)).to(device)
        target_soft = renderer.render(true_shape_t, true_dy_t, true_eps_t)
        soft_metrics = renderer.eval_soft_metrics(pred_soft, target_soft)
        soft_sanity = renderer.eval_soft_metrics(target_soft, target_soft)

    cls_metrics = accuracy_f1(shapes_true, shape_pred)

    case_rows = []
    for pos, case_idx in enumerate(idx_test):
        case_level = obstacle_iou_and_dice(
            targets[pos : pos + 1],
            predictions[pos : pos + 1],
            threshold=threshold,
        )
        case_rows.append(
            {
                "case_id": bundle.case_ids[case_idx],
                "shape_true": shapes_true[pos],
                "shape_pred": shape_pred[pos],
                "shape_correct": bool(shape_pred[pos] == shapes_true[pos]),
                "Re": int(bundle.re_values[case_idx]),
                "dy_true": float(dy_true[pos]),
                "dy_pred": float(dy_pred[pos]),
                "dy_abs_err": float(abs(dy_pred[pos] - dy_true[pos])),
                "eps_true": float(eps_true[pos]),
                "eps_pred": float(eps_pred[pos]),
                "eps_abs_err": float(abs(eps_pred[pos] - eps_true[pos])),
                "inverse_iou": float(case_level["iou_values"][0]),
                "inverse_dice": float(case_level["dice_values"][0]),
            }
        )

    case_df = pd.DataFrame(case_rows).sort_values(["shape_true", "case_id"]).reset_index(drop=True)
    case_df.to_csv(reports_dir / "wake_field_reconstruction_cases.csv", index=False)

    summary_metrics = {
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "dy_mae": float(np.mean(np.abs(dy_pred - dy_true))),
        "eps_mae": float(np.mean(np.abs(eps_pred - eps_true))),
        "inverse_iou": inv_metrics["iou_mean"],
        "inverse_dice": inv_metrics["dice_mean"],
        "soft_iou": soft_metrics["soft_iou"],
        "soft_dice": soft_metrics["soft_dice"],
    }
    sanity_summary = {
        "inverse_iou": sanity_metrics["iou_mean"],
        "inverse_dice": sanity_metrics["dice_mean"],
        "soft_iou": soft_sanity["soft_iou"],
        "soft_dice": soft_sanity["soft_dice"],
    }

    _plot_examples(
        case_ids=bundle.case_ids[idx_test],
        targets=targets,
        predictions=predictions,
        shape_true=shapes_true,
        shape_pred=shape_pred,
        output_path=reports_dir / "wake_field_reconstruction_examples.png",
    )
    _write_summary(
        output_path=reports_dir / "wake_field_reconstruction_summary.md",
        variant_name=variant_name,
        metrics=summary_metrics,
        sanity_metrics=sanity_summary,
    )

    logger.info(
        "Wake-field reconstruction done. acc=%.4f macro_f1=%.4f iou=%.4f dice=%.4f report=%s",
        summary_metrics["accuracy"],
        summary_metrics["macro_f1"],
        summary_metrics["inverse_iou"],
        summary_metrics["inverse_dice"],
        reports_dir / "wake_field_reconstruction_summary.md",
    )


if __name__ == "__main__":
    main()
