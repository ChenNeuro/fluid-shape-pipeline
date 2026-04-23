from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sim.data_schema import WAKE_FIELD_FILENAME, find_wake_frames_npz, read_metadata, write_metadata


SUPPORTED_CHANNELS = ("ux", "uy", "speed", "vorticity")


def _ensure_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError(f"Expected wake frames with shape [T, H, W], got {frames.shape}")
    if frames.shape[0] < 2:
        raise ValueError("Need at least two frames to estimate optical flow")
    return frames.astype(np.float32)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.round(frame * 255.0).astype(np.uint8)


def _estimate_pairwise_flow(frames: np.ndarray, estimator: str) -> np.ndarray:
    if estimator != "farneback":
        raise ValueError(f"Unsupported flow estimator: {estimator}")

    flows = []
    for idx in range(frames.shape[0] - 1):
        prev = _to_uint8(frames[idx])
        nxt = _to_uint8(frames[idx + 1])
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flows.append(flow.astype(np.float32))
    return np.stack(flows, axis=0)


def _derive_field(flows: np.ndarray, channel_names: list[str]) -> np.ndarray:
    flow_mean = flows.mean(axis=0)
    ux = flow_mean[..., 0]
    uy = flow_mean[..., 1]
    speed = np.sqrt(ux**2 + uy**2)
    vorticity = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    channel_map = {
        "ux": ux,
        "uy": uy,
        "speed": speed,
        "vorticity": vorticity,
    }
    return np.stack([channel_map[name] for name in channel_names], axis=0).astype(np.float32)


def _hotspot_box(vorticity: np.ndarray) -> list[float]:
    height, width = vorticity.shape
    search_w = max(1, int(round(0.40 * width)))
    crop_w = max(2, int(round(0.25 * width)))
    crop_h = max(2, int(round(0.35 * height)))

    window = np.abs(vorticity[:, :search_w])
    y_idx, x_idx = np.unravel_index(int(np.argmax(window)), window.shape)

    x0 = int(np.clip(x_idx - crop_w // 2, 0, width - crop_w))
    y0 = int(np.clip(y_idx - crop_h // 2, 0, height - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def build_crop_boxes(vorticity: np.ndarray, scales: list[str]) -> dict[str, list[float]]:
    boxes: dict[str, list[float]] = {}
    for scale in scales:
        if scale == "full":
            boxes[scale] = [0.0, 0.0, 1.0, 1.0]
        elif scale == "half":
            boxes[scale] = [0.0, 0.0, 0.5, 1.0]
        elif scale == "quarter":
            boxes[scale] = [0.0, 0.25, 0.5, 0.75]
        elif scale == "hotspot":
            boxes[scale] = _hotspot_box(vorticity)
        else:
            raise ValueError(f"Unsupported wake scale: {scale}")
    return boxes


def _slice_bounds(norm_box: list[float], height: int, width: int) -> tuple[int, int, int, int]:
    x0 = int(np.floor(norm_box[0] * width))
    y0 = int(np.floor(norm_box[1] * height))
    x1 = int(np.ceil(norm_box[2] * width))
    y1 = int(np.ceil(norm_box[3] * height))

    x0 = int(np.clip(x0, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    x1 = int(np.clip(max(x1, x0 + 1), x0 + 1, width))
    y1 = int(np.clip(max(y1, y0 + 1), y0 + 1, height))
    return x0, y0, x1, y1


def resize_crop(field: np.ndarray, norm_box: list[float], output_size: int) -> np.ndarray:
    _, height, width = field.shape
    x0, y0, x1, y1 = _slice_bounds(norm_box, height=height, width=width)
    crop = field[:, y0:y1, x0:x1]
    resized = [
        cv2.resize(channel, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        for channel in crop
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def normalize_field(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_mean = field.reshape(field.shape[0], -1).mean(axis=1).astype(np.float32)
    channel_std = field.reshape(field.shape[0], -1).std(axis=1).astype(np.float32)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    normalized = (field - channel_mean[:, None, None]) / channel_std[:, None, None]
    return normalized.astype(np.float32), channel_mean, channel_std


def build_case_wake_field(case_dir: Path, cfg: dict) -> dict:
    vision_cfg = cfg.get("vision", {})
    estimator = str(vision_cfg.get("flow_estimator", "farneback"))
    field_size = int(vision_cfg.get("field_size", 128))
    channel_names = [str(name) for name in vision_cfg.get("channels", SUPPORTED_CHANNELS)]
    scales = [str(name) for name in vision_cfg.get("scales", ["full", "half", "quarter", "hotspot"])]

    for channel_name in channel_names:
        if channel_name not in SUPPORTED_CHANNELS:
            raise ValueError(f"Unsupported wake field channel: {channel_name}")

    frames_payload = np.load(find_wake_frames_npz(case_dir))
    frames = _ensure_frames(frames_payload["frames"])
    pairwise_flows = _estimate_pairwise_flow(frames, estimator=estimator)
    field_raw = _derive_field(pairwise_flows, channel_names=channel_names)
    field_norm, channel_mean, channel_std = normalize_field(field_raw)

    vorticity_idx = channel_names.index("vorticity")
    crop_boxes = build_crop_boxes(field_raw[vorticity_idx], scales=scales)
    crops = np.stack([resize_crop(field_norm, crop_boxes[scale], output_size=field_size) for scale in scales], axis=0)

    wake_field_path = case_dir / WAKE_FIELD_FILENAME
    np.savez_compressed(
        wake_field_path,
        field_raw=field_raw,
        field_norm=field_norm,
        crops=crops,
        scales=np.asarray(scales),
        crop_boxes=np.asarray([crop_boxes[scale] for scale in scales], dtype=np.float32),
        channel_names=np.asarray(channel_names),
        channel_mean=channel_mean,
        channel_std=channel_std,
        source_frames=np.asarray(frames.shape[0], dtype=np.int32),
        flow_pair_count=np.asarray(pairwise_flows.shape[0], dtype=np.int32),
    )

    metadata = read_metadata(case_dir)
    metadata["field_channels"] = channel_names
    metadata["crop_boxes"] = crop_boxes
    metadata.setdefault("files", {})
    metadata["files"]["wake_field_npz"] = WAKE_FIELD_FILENAME
    write_metadata(case_dir, metadata)

    wake_roi = metadata.get("wake_roi", {})
    row = {
        "case_id": str(metadata["case_id"]),
        "shape": str(metadata["shape"]),
        "Re": int(metadata["Re"]),
        "dy": float(metadata["dy"]),
        "eps": float(metadata["eps"]),
        "seed": int(metadata["seed"]),
        "wake_field_npz": str(wake_field_path),
        "wake_frames": int(frames.shape[0]),
        "field_size": int(field_size),
        "channels": "|".join(channel_names),
        "scales": "|".join(scales),
        "flow_pair_count": int(pairwise_flows.shape[0]),
        "wake_x_min": float(wake_roi.get("x_min", 0.0)),
        "wake_x_max": float(wake_roi.get("x_max", 0.0)),
        "wake_y_min": float(wake_roi.get("y_min", 0.0)),
        "wake_y_max": float(wake_roi.get("y_max", 0.0)),
    }
    for scale, box in crop_boxes.items():
        row[f"{scale}_box"] = "|".join(f"{value:.6f}" for value in box)
    return row
