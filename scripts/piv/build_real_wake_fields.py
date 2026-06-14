from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from sim.data_schema import WAKE_FIELD_FILENAME
from vision.wake_field_builder import normalize_field, resize_crop

CHANNELS = ["ux", "uy", "speed", "vorticity"]
SCALES = ["distD1.0_full", "distD2.0_full", "distD4.0_full"]
MODEL_FRACTION_CROP_BOXES = {
    "distD1.0_full": [0.142452, 0.0, 1.0, 1.0],
    "distD2.0_full": [0.284905, 0.0, 1.0, 1.0],
    "distD4.0_full": [0.569810, 0.0, 1.0, 1.0],
}
ROI_FRACTION_CROP_BOXES = {
    "distD1.0_full": [0.0, 0.0, 1.0, 1.0],
    "distD2.0_full": [0.25, 0.0, 1.0, 1.0],
    "distD4.0_full": [0.50, 0.0, 1.0, 1.0],
}
SHAPE_MAP = {
    "\u5706": "circle",
    "\u4e09\u89d2": "triangle",
    "\u673a\u7ffc": "airfoil",
    "\u83f1\u5f62": "diamond",
    "\u957f\u65b9": "bar",
}


@dataclass(frozen=True)
class PivGroup:
    directory: Path
    source_name: str
    shape_cn: str
    shape: str
    speed_level: int
    sequence: int
    csv_files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wake_field.npz files from real PIV CSVs")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("\u84dd\u6708\u4f20\u5947") / "\u8ba1\u7b97\u7ed3\u679c",
        help="Root containing shape-speed-sequence CSV directories",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=Path("/home/chenyihao/fluid_runs/piv_blueluna_validation"),
        help="Output run directory; prefer WSL home to avoid filling C:",
    )
    parser.add_argument("--field-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth CSV in each sequence")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--water-height-m", type=float, default=0.35)
    parser.add_argument("--camera-y-m", type=float, default=0.175)
    parser.add_argument("--tank-width-m", type=float, default=0.40)
    parser.add_argument(
        "--equivalent-diameter-m", type=float, default=math.sqrt(4.0 * 0.005 / math.pi)
    )
    parser.add_argument("--nu-m2-s", type=float, default=1.0e-6)
    parser.add_argument(
        "--assigned-re",
        type=int,
        default=800,
        help="Training Re class assigned to all real PIV rows unless --re-by-level is used",
    )
    parser.add_argument(
        "--re-by-level",
        default="",
        help="Optional mapping like 5:500,10:800,15:1100. Empty means --assigned-re for all.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["model_fractions", "roi_fractions"],
        default="model_fractions",
        help="model_fractions matches CFD crop fractions; roi_fractions uses the visible PIV ROI.",
    )
    parser.add_argument(
        "--max-per-sequence",
        type=int,
        default=0,
        help="Optional cap after stride. 0 keeps all selected CSVs.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip PIV vertical coordinate and negate V; useful for image-origin exports.",
    )
    return parser.parse_args()


def _numeric_stem(path: Path) -> int:
    return int(path.stem) if path.stem.isdigit() else 10**9


def _parse_group(directory: Path) -> PivGroup | None:
    parts = directory.name.strip().split("-")
    if len(parts) != 3:
        return None
    shape_cn = parts[0].strip()
    speed_match = re.search(r"\d+", parts[1])
    sequence_match = re.search(r"\d+", parts[2])
    if speed_match is None or sequence_match is None:
        return None
    shape = SHAPE_MAP.get(shape_cn)
    if shape is None:
        return None
    csv_files = sorted(directory.glob("*.csv"), key=_numeric_stem)
    return PivGroup(
        directory=directory,
        source_name=directory.name,
        shape_cn=shape_cn,
        shape=shape,
        speed_level=int(speed_match.group()),
        sequence=int(sequence_match.group()),
        csv_files=csv_files,
    )


def discover_groups(source_root: Path) -> list[PivGroup]:
    if not source_root.exists():
        raise FileNotFoundError(f"PIV source root not found: {source_root}")
    groups = []
    for directory in sorted([path for path in source_root.iterdir() if path.is_dir()]):
        group = _parse_group(directory)
        if group is not None:
            groups.append(group)
    if not groups:
        raise RuntimeError(f"No PIV CSV groups parsed from {source_root}")
    return groups


def read_piv_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[0, 2])
    frame = frame.loc[:, [col for col in frame.columns if str(col).strip()]]
    frame.columns = [str(col).strip() for col in frame.columns]
    required = [
        "X(mm)",
        "Y(mm)",
        "Velocity |V|(mm/s)",
        "Velocity U(mm/s)",
        "Velocity V(mm/s)",
        "Correlation Value",
        "Flag",
        "Rotation Tensor(rad)",
        "Peak Ratio",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing PIV CSV columns in {path}: {missing}")
    return frame


def _grid(
    values: pd.Series, x_values: np.ndarray, y_values: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    pivot = frame.assign(_value=values).pivot_table(
        index="Y(mm)", columns="X(mm)", values="_value", aggfunc="mean"
    )
    pivot = pivot.reindex(index=y_values, columns=x_values)
    array = pivot.to_numpy(dtype=np.float64)
    if np.isnan(array).any():
        array = np.where(np.isnan(array), np.nanmedian(array), array)
    return array.astype(np.float32)


def csv_to_field(frame: pd.DataFrame, *, flip_y: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    x_values = np.sort(frame["X(mm)"].unique())
    y_values = np.sort(frame["Y(mm)"].unique())
    ux_mm_s = _grid(frame["Velocity U(mm/s)"], x_values, y_values, frame)
    uy_mm_s = _grid(frame["Velocity V(mm/s)"], x_values, y_values, frame)
    if flip_y:
        ux_mm_s = ux_mm_s[::-1, :]
        uy_mm_s = -uy_mm_s[::-1, :]

    x_m = x_values.astype(np.float64) * 1.0e-3
    y_m = y_values.astype(np.float64) * 1.0e-3
    ux = ux_mm_s * 1.0e-3
    uy = uy_mm_s * 1.0e-3
    speed = np.sqrt(ux**2 + uy**2)
    d_uy_dx = np.gradient(uy, x_m, axis=1)
    d_ux_dy = np.gradient(ux, y_m, axis=0)
    vorticity = d_uy_dx - d_ux_dy
    field = np.stack([ux, uy, speed, vorticity], axis=0).astype(np.float32)

    metadata = {
        "grid_nx": int(len(x_values)),
        "grid_ny": int(len(y_values)),
        "x_min_mm": float(x_values.min()),
        "x_max_mm": float(x_values.max()),
        "y_min_mm": float(y_values.min()),
        "y_max_mm": float(y_values.max()),
        "correlation_mean": float(frame["Correlation Value"].mean()),
        "peak_ratio_median": float(frame["Peak Ratio"].median()),
        "flag_mean": float(frame["Flag"].mean()),
        "rotation_tensor_mean": float(frame["Rotation Tensor(rad)"].mean()),
    }
    return field, metadata


def estimate_freestream_mm_s(frame: pd.DataFrame) -> float:
    y_values = np.sort(frame["Y(mm)"].unique())
    low_cut = y_values[max(0, int(round(0.20 * (len(y_values) - 1))))]
    high_cut = y_values[min(len(y_values) - 1, int(round(0.80 * (len(y_values) - 1))))]
    border = frame[(frame["Y(mm)"] <= low_cut) | (frame["Y(mm)"] >= high_cut)]
    if border.empty:
        border = frame
    return float(border["Velocity U(mm/s)"].median())


def _resize_field(field: np.ndarray, output_size: int) -> np.ndarray:
    resized = [
        cv2.resize(channel, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        for channel in field
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def _parse_re_by_level(mapping: str, assigned_re: int) -> dict[int, int]:
    if not mapping:
        return {}
    result = {}
    for part in mapping.split(","):
        key, value = part.split(":", 1)
        result[int(key.strip())] = int(value.strip())
    return result or {0: int(assigned_re)}


def _selected_csvs(group: PivGroup, stride: int, max_per_sequence: int) -> list[Path]:
    selected = group.csv_files[:: max(1, int(stride))]
    if max_per_sequence > 0:
        selected = selected[: int(max_per_sequence)]
    return selected


def _case_id(group: PivGroup, csv_path: Path) -> str:
    return (
        f"piv_{group.shape}_lvl{group.speed_level:02d}_"
        f"seq{group.sequence:02d}_t{_numeric_stem(csv_path):04d}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    run_dir = args.output_run_dir.expanduser().resolve()
    raw_dir = run_dir / "data" / "raw"
    wake_dir = run_dir / "data" / "wake_fields"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wake_dir.mkdir(parents=True, exist_ok=True)

    re_by_level = _parse_re_by_level(args.re_by_level, args.assigned_re)
    crop_boxes = (
        MODEL_FRACTION_CROP_BOXES
        if args.crop_mode == "model_fractions"
        else ROI_FRACTION_CROP_BOXES
    )
    groups = discover_groups(source_root)
    rows = []
    manifest_rows = []
    group_rows = []
    skipped = 0
    for group in groups:
        selected = _selected_csvs(group, args.stride, args.max_per_sequence)
        freestream_values = []
        for csv_path in selected:
            frame = read_piv_csv(csv_path)
            field_raw_native, csv_meta = csv_to_field(frame, flip_y=bool(args.flip_y))
            field_raw = _resize_field(field_raw_native, int(args.field_size))
            field_norm, channel_mean, channel_std = normalize_field(field_raw)
            crops = np.stack(
                [
                    resize_crop(field_norm, crop_boxes[scale], output_size=int(args.field_size))
                    for scale in SCALES
                ],
                axis=0,
            )

            u_free_mm_s = estimate_freestream_mm_s(frame)
            estimated_re = (
                u_free_mm_s
                * float(args.equivalent_diameter_m)
                * 1000.0
                / (float(args.nu_m2_s) * 1.0e6)
            )
            freestream_values.append(u_free_mm_s)
            assigned_re = int(re_by_level.get(group.speed_level, args.assigned_re))

            case_id = _case_id(group, csv_path)
            case_dir = raw_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            wake_field_path = case_dir / WAKE_FIELD_FILENAME
            np.savez_compressed(
                wake_field_path,
                field_raw=field_raw,
                field_norm=field_norm,
                crops=crops.astype(np.float32),
                scales=np.asarray(SCALES),
                crop_boxes=np.asarray([crop_boxes[scale] for scale in SCALES], dtype=np.float32),
                channel_names=np.asarray(CHANNELS),
                channel_mean=channel_mean,
                channel_std=channel_std,
                source_frames=np.asarray(0, dtype=np.int32),
                flow_pair_count=np.asarray(0, dtype=np.int32),
                freestream_u_mm_s=np.asarray(u_free_mm_s, dtype=np.float32),
                estimated_re=np.asarray(estimated_re, dtype=np.float32),
            )

            local_x_min = float(csv_meta["x_min_mm"]) * 1.0e-3
            local_x_max = float(csv_meta["x_max_mm"]) * 1.0e-3
            local_y_min = float(csv_meta["y_min_mm"]) * 1.0e-3
            local_y_max = float(csv_meta["y_max_mm"]) * 1.0e-3
            metadata = {
                "case_id": case_id,
                "source_case_id": f"{group.shape}_lvl{group.speed_level}_seq{group.sequence}",
                "backend": "piv_csv",
                "domain": "piv",
                "shape": group.shape,
                "shape_cn": group.shape_cn,
                "speed_level": group.speed_level,
                "Re": assigned_re,
                "estimated_re": estimated_re,
                "freestream_u_mm_s": u_free_mm_s,
                "dy": 0.0,
                "eps": 0.0,
                "seed": group.sequence,
                "split": "validation",
                "fps": float(args.fps),
                "csv_index": _numeric_stem(csv_path),
                "source_csv": str(csv_path),
                "source_group": group.source_name,
                "mount_note": (
                    "airfoil mount reported around half-chord, not centroid"
                    if group.shape == "airfoil"
                    else "centered as planned"
                ),
                "experiment": {
                    "water_height_m": float(args.water_height_m),
                    "camera_y_m": float(args.camera_y_m),
                    "tank_width_m": float(args.tank_width_m),
                    "flow_meter": "broken; speed inferred from PIV boundary velocity",
                },
                "csv_grid": csv_meta,
                "wake_roi": {
                    "coordinate_frame": "local_piv_export_mm",
                    "x_min": local_x_min,
                    "x_max": local_x_max,
                    "y_min": local_y_min,
                    "y_max": local_y_max,
                    "width": local_x_max - local_x_min,
                    "height": local_y_max - local_y_min,
                    "pixels_x": int(args.field_size),
                    "pixels_y": int(args.field_size),
                },
                "field_channels": CHANNELS,
                "crop_boxes": crop_boxes,
                "crop_mode": args.crop_mode,
                "flip_y": bool(args.flip_y),
                "files": {"wake_field_npz": WAKE_FIELD_FILENAME, "source_csv": str(csv_path)},
            }
            _write_json(case_dir / "metadata.json", metadata)

            row = {
                "case_id": case_id,
                "source_case_id": metadata["source_case_id"],
                "shape": group.shape,
                "shape_cn": group.shape_cn,
                "speed_level": group.speed_level,
                "sequence": group.sequence,
                "Re": assigned_re,
                "estimated_re": estimated_re,
                "freestream_u_mm_s": u_free_mm_s,
                "dy": 0.0,
                "eps": 0.0,
                "seed": group.sequence,
                "split": "validation",
                "domain": "piv",
                "csv_index": _numeric_stem(csv_path),
                "source_csv": str(csv_path),
                "wake_field_npz": str(wake_field_path),
                "wake_frames": 0,
                "field_size": int(args.field_size),
                "native_grid_nx": csv_meta["grid_nx"],
                "native_grid_ny": csv_meta["grid_ny"],
                "channels": "|".join(CHANNELS),
                "scales": "|".join(SCALES),
                "flow_pair_count": 0,
                "canvas_x_start": local_x_min,
                "canvas_x_end": local_x_max,
                "canvas_y_min": local_y_min,
                "canvas_y_max": local_y_max,
                "crop_mode": args.crop_mode,
                "flip_y": bool(args.flip_y),
                "correlation_mean": csv_meta["correlation_mean"],
                "peak_ratio_median": csv_meta["peak_ratio_median"],
                "flag_mean": csv_meta["flag_mean"],
            }
            for scale, box in crop_boxes.items():
                row[f"{scale}_box"] = "|".join(f"{value:.6f}" for value in box)
            rows.append(row)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "status": "success",
                    "source_csv": str(csv_path),
                    "wake_field_npz": str(wake_field_path),
                }
            )

        if selected:
            group_rows.append(
                {
                    "directory": str(group.directory),
                    "source_name": group.source_name,
                    "shape_cn": group.shape_cn,
                    "shape": group.shape,
                    "speed_level": group.speed_level,
                    "sequence": group.sequence,
                    "csv_files": len(group.csv_files),
                    "selected_csvs": len(selected),
                    "freestream_u_median_mm_s": float(np.median(freestream_values)),
                    "estimated_re_median": float(
                        np.median(freestream_values)
                        * float(args.equivalent_diameter_m)
                        * 1000.0
                        / (float(args.nu_m2_s) * 1.0e6)
                    ),
                }
            )
        else:
            skipped += 1

    if not rows:
        raise RuntimeError("No real PIV wake fields were built")

    index_df = pd.DataFrame(rows).sort_values("case_id")
    index_df.to_csv(wake_dir / "index.csv", index=False)
    pd.DataFrame(manifest_rows).sort_values("case_id").to_csv(raw_dir / "manifest.csv", index=False)
    group_df = pd.DataFrame(group_rows).sort_values(["shape", "speed_level", "sequence"])
    group_df.to_csv(wake_dir / "source_groups.csv", index=False)
    speed_summary = (
        group_df.groupby("speed_level")
        .agg(
            groups=("source_name", "count"),
            selected_csvs=("selected_csvs", "sum"),
            freestream_u_median_mm_s=("freestream_u_median_mm_s", "median"),
            estimated_re_median=("estimated_re_median", "median"),
        )
        .reset_index()
    )
    speed_summary.to_csv(wake_dir / "speed_level_summary.csv", index=False)
    summary = {
        "rows": len(rows),
        "groups": len(groups),
        "skipped_groups": skipped,
        "source_root": str(source_root),
        "index_csv": str(wake_dir / "index.csv"),
        "field_size": int(args.field_size),
        "stride": int(args.stride),
        "crop_mode": args.crop_mode,
        "flip_y": bool(args.flip_y),
        "assigned_re": int(args.assigned_re),
        "re_by_level": args.re_by_level,
        "notes": [
            "Real PIV CSV coordinate frame is local to exported camera ROI.",
            "Flow meter was broken; estimated_re uses top/bottom boundary-strip median U.",
            "All rows are a validation split; do not mix frames from the same sequence.",
        ],
    }
    _write_json(wake_dir / "summary.json", summary)
    print(
        "Built "
        f"{len(rows)} real PIV wake fields from {len(groups)} groups; "
        f"index={wake_dir / 'index.csv'}"
    )
    print(speed_summary.to_string(index=False))


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
