from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sim.config import repo_root


@dataclass(frozen=True)
class WakeBundle:
    case_ids: np.ndarray
    shapes: np.ndarray
    re_values: np.ndarray
    dy: np.ndarray
    eps: np.ndarray
    seeds: np.ndarray
    scale_names: list[str]
    channel_names: list[str]
    crops_by_scale: dict[str, np.ndarray]


def load_wake_bundle(wake_fields_dir: str | Path | None = None) -> WakeBundle:
    if wake_fields_dir is None:
        index_path = repo_root() / "data" / "wake_fields" / "index.csv"
    else:
        index_path = Path(wake_fields_dir).expanduser().resolve() / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing wake-field index: {index_path}. Run build_wake_fields first."
        )

    index_df = pd.read_csv(index_path).sort_values("case_id").reset_index(drop=True)
    if index_df.empty:
        raise RuntimeError("Wake-field index is empty")

    crops_by_scale: dict[str, list[np.ndarray]] = {}
    scale_names: list[str] | None = None
    channel_names: list[str] | None = None

    for _, row in index_df.iterrows():
        payload = np.load(str(row["wake_field_npz"]))
        case_scales = [str(item) for item in payload["scales"].tolist()]
        case_channels = [str(item) for item in payload["channel_names"].tolist()]
        crops = payload["crops"].astype(np.float32)

        if scale_names is None:
            scale_names = case_scales
        elif case_scales != scale_names:
            raise RuntimeError(
                f"Inconsistent scale names for case {row['case_id']}: "
                f"{case_scales} != {scale_names}"
            )

        if channel_names is None:
            channel_names = case_channels
        elif case_channels != channel_names:
            raise RuntimeError(
                f"Inconsistent channel names for case {row['case_id']}: "
                f"{case_channels} != {channel_names}"
            )

        for idx, scale in enumerate(case_scales):
            crops_by_scale.setdefault(scale, []).append(crops[idx])

    if scale_names is None or channel_names is None:
        raise RuntimeError("Failed to infer wake scale/channel names from wake-field index")

    stacked = {
        scale: np.stack(items, axis=0).astype(np.float32) for scale, items in crops_by_scale.items()
    }
    return WakeBundle(
        case_ids=index_df["case_id"].to_numpy(dtype=str),
        shapes=index_df["shape"].to_numpy(dtype=str),
        re_values=index_df["Re"].to_numpy(dtype=int),
        dy=index_df["dy"].to_numpy(dtype=np.float32),
        eps=index_df["eps"].to_numpy(dtype=np.float32),
        seeds=index_df["seed"].to_numpy(dtype=int),
        scale_names=scale_names,
        channel_names=channel_names,
        crops_by_scale=stacked,
    )


def variant_tensor(
    bundle: WakeBundle,
    *,
    scales: Sequence[str],
    channels: Sequence[str],
) -> np.ndarray:
    missing_scales = [scale for scale in scales if scale not in bundle.crops_by_scale]
    if missing_scales:
        raise ValueError(f"Requested scales not found in wake bundle: {missing_scales}")

    channel_to_idx = {name: idx for idx, name in enumerate(bundle.channel_names)}
    try:
        channel_idx = [channel_to_idx[name] for name in channels]
    except KeyError as exc:
        raise ValueError(f"Requested channel not found in wake bundle: {exc}") from exc

    stacked = np.stack(
        [bundle.crops_by_scale[scale][:, channel_idx, :, :] for scale in scales], axis=1
    )
    return stacked.astype(np.float32)
