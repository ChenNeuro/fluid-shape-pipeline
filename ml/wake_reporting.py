from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_variant_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
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


def plot_training_curves(
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
        ax.plot(epochs, losses, marker="o", label=f"{variant_name} train")
        val_losses = [row.get("val_loss") for row in history]
        if any(v is not None for v in val_losses):
            val_epochs = [row["epoch"] for row in history if row.get("val_loss") is not None]
            val_vals = [row["val_loss"] for row in history if row.get("val_loss") is not None]
            ax.plot(val_epochs, val_vals, marker="s", linestyle="--", label=f"{variant_name} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Wake-Field Training Curves (seed={export_seed})")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(
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


def write_wake_summary(
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
        (
            f"- Main repeated holdout: acc={main_row['accuracy_mean']:.4f}+/-"
            f"{main_row['accuracy_std']:.4f}, macroF1={main_row['macro_f1_mean']:.4f}+/-"
            f"{main_row['macro_f1_std']:.4f}"
        ),
        (
            f"- Single-scale (`{single_variant}`) vs multi-scale macroF1: "
            f"{single_row['macro_f1_mean']:.4f} -> {main_row['macro_f1_mean']:.4f}"
        ),
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
            f"- Re={int(row['Re_test'])}: acc={row['accuracy']:.4f}, "
            f"macroF1={row['macro_f1']:.4f}, dy_MAE={row['dy_mae']:.5f}, "
            f"eps_MAE={row['eps_mae']:.5f}, IoU={row['inverse_iou']:.4f}, "
            f"Dice={row['inverse_dice']:.4f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
