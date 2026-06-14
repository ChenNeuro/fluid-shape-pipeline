from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from scripts.cfd.common import cfd_geometry, parse_float_line, read_config, read_json, write_json
from sim.data_schema import WAKE_FIELD_FILENAME
from vision.wake_field_builder import build_crop_boxes, normalize_field, resize_crop

SUPPORTED_CHANNELS = ("ux", "uy", "speed", "vorticity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build wake_field.npz tensors from OpenFOAM velocity fields"
    )
    parser.add_argument("--config", required=True, help="CFD YAML config")
    parser.add_argument("--case-root", required=True, help="OpenFOAM case root")
    parser.add_argument("--run-dir", required=True, help="Wake dataset run directory")
    parser.add_argument(
        "--openfoam-bashrc",
        default="/usr/share/openfoam/etc/bashrc",
        help="OpenFOAM bashrc for optional postProcess fallback",
    )
    parser.add_argument(
        "--time-mode",
        choices=["latest", "all"],
        default="latest",
        help="Use only the latest OpenFOAM time or every written time directory.",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=0.0,
        help="Minimum OpenFOAM time to include when --time-mode all is used.",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=None,
        help="Minimum convective time tau=tU/D to include when --time-mode all is used.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=None,
        help="Maximum convective time tau=tU/D to include when --time-mode all is used.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel case workers")
    return parser.parse_args()


def _source_prefix(openfoam_bashrc: str) -> str:
    if Path(openfoam_bashrc).exists():
        return f"source {openfoam_bashrc} >/tmp/of_source.log 2>&1 || true; "
    return ""


def _run_write_cell_centres(case_dir: Path, openfoam_bashrc: str) -> None:
    command = _source_prefix(openfoam_bashrc) + "postProcess -func writeCellCentres -latestTime"
    subprocess.run(["bash", "-lc", command], cwd=case_dir, check=True)


def _parse_time_dir(path: Path) -> float:
    try:
        return float(path.name)
    except ValueError:
        return -1.0


def _numeric_time_dirs(case_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in case_dir.iterdir()
            if path.is_dir()
            and path.name not in {"0", "constant", "system", "logs", "postProcessing"}
            and _parse_time_dir(path) >= 0.0
        ],
        key=_parse_time_dir,
    )


def _latest_time_dir(case_dir: Path) -> Path:
    dirs = _numeric_time_dirs(case_dir)
    if not dirs:
        raise RuntimeError(f"No numeric OpenFOAM time directories found in {case_dir}")
    return dirs[-1]


def _selected_time_dirs(
    case_dir: Path,
    *,
    time_mode: str,
    time_min: float,
    tau_min: float | None = None,
    tau_max: float | None = None,
) -> list[Path | None]:
    if time_mode == "latest":
        return [None]
    meta = read_json(case_dir / "case_metadata.json")
    case = meta["case"]
    geometry = meta["geometry"]
    inlet_u = float(case["inlet_u_m_s"])
    obstacle_d = float(geometry["equivalent_diameter_m"])
    dirs = []
    for path in _numeric_time_dirs(case_dir):
        time_value = _parse_time_dir(path)
        tau_value = time_value * inlet_u / obstacle_d
        if time_value < time_min:
            continue
        if tau_min is not None and tau_value < tau_min:
            continue
        if tau_max is not None and tau_value > tau_max:
            continue
        dirs.append(path)
    if not dirs:
        raise RuntimeError(
            f"No OpenFOAM time directories matched time_min={time_min}, "
            f"tau_min={tau_min}, tau_max={tau_max} in {case_dir}"
        )
    return dirs


def _time_tag(time_value: float) -> str:
    milliseconds = int(round(time_value * 1000.0))
    return f"t{milliseconds:07d}ms"


def _sample_files(case_dir: Path) -> list[Path]:
    root = case_dir / "postProcessing"
    if not root.exists():
        return []
    patterns = ["**/*wakePlane*U*.raw", "**/*wakePlane*U*.xy", "**/*U*.raw", "**/*U*.xy"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files), key=lambda path: path.stat().st_mtime)


def _load_sample_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = parse_float_line(stripped)
        if len(values) >= 6:
            rows.append([values[0], values[1], values[2], values[3], values[4]])
        elif len(values) >= 5:
            rows.append([values[0], values[1], 0.0, values[3], values[4]])
    if not rows:
        raise RuntimeError(f"No OpenFOAM sample rows parsed from {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 3], data[:, 4]


def _read_internal_vectors(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if "internalField" not in line:
            continue
        if "uniform" in line and "nonuniform" not in line:
            values = parse_float_line(line)
            if len(values) >= 3:
                return np.asarray([values[-3:]], dtype=np.float64)
        if "nonuniform" not in line:
            continue
        count_idx = idx + 1
        while count_idx < len(lines) and not lines[count_idx].strip().isdigit():
            count_idx += 1
        if count_idx >= len(lines):
            break
        count = int(lines[count_idx].strip())
        start_idx = count_idx + 1
        while start_idx < len(lines) and lines[start_idx].strip() != "(":
            start_idx += 1
        vectors = []
        for value_line in lines[start_idx + 1 : start_idx + 1 + count]:
            values = parse_float_line(value_line)
            if len(values) >= 3:
                vectors.append(values[:3])
        if len(vectors) != count:
            raise RuntimeError(f"Expected {count} vectors in {path}, parsed {len(vectors)}")
        return np.asarray(vectors, dtype=np.float64)
    raise RuntimeError(f"No internal vector field found in {path}")


def _load_cell_centre_points(
    case_dir: Path, openfoam_bashrc: str, time_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latest_dir = _latest_time_dir(case_dir)
    if not (latest_dir / "C").exists():
        _run_write_cell_centres(case_dir, openfoam_bashrc)
    velocity_dir = latest_dir if time_dir is None else time_dir
    centres = _read_internal_vectors(latest_dir / "C")
    velocities = _read_internal_vectors(velocity_dir / "U")
    if velocities.shape[0] == 1 and centres.shape[0] > 1:
        velocities = np.repeat(velocities, centres.shape[0], axis=0)
    if centres.shape[0] != velocities.shape[0]:
        raise RuntimeError(
            "Cell centre count and U count differ in "
            f"{velocity_dir}: {centres.shape[0]} vs {velocities.shape[0]}"
        )
    return centres[:, 0], centres[:, 1], velocities[:, 0], velocities[:, 1]


def _interpolate_field(
    *,
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    points = np.stack([x, y], axis=1)
    ux_grid = griddata(points, ux, (grid_x, grid_y), method="linear")
    uy_grid = griddata(points, uy, (grid_x, grid_y), method="linear")
    if np.isnan(ux_grid).any():
        ux_grid = np.where(
            np.isnan(ux_grid),
            griddata(points, ux, (grid_x, grid_y), method="nearest"),
            ux_grid,
        )
    if np.isnan(uy_grid).any():
        uy_grid = np.where(
            np.isnan(uy_grid),
            griddata(points, uy, (grid_x, grid_y), method="nearest"),
            uy_grid,
        )
    return ux_grid.astype(np.float32), uy_grid.astype(np.float32)


def _derive_field(
    ux: np.ndarray, uy: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray
) -> np.ndarray:
    speed = np.sqrt(ux**2 + uy**2)
    d_uy_dx = np.gradient(uy, x_coords, axis=1)
    d_ux_dy = np.gradient(ux, y_coords, axis=0)
    vorticity = d_uy_dx - d_ux_dy
    return np.stack([ux, uy, speed, vorticity], axis=0).astype(np.float32)


def _case_dirs(case_root: Path) -> list[Path]:
    return sorted(path for path in case_root.iterdir() if (path / "case_metadata.json").exists())


def _status_for_case(case_id: str, health_df: pd.DataFrame | None) -> str:
    if health_df is None or health_df.empty or "case_id" not in health_df:
        return "unknown"
    rows = health_df[health_df["case_id"] == case_id]
    if rows.empty:
        return "unknown"
    return str(rows.iloc[0].get("status", "unknown"))


def _load_health(case_root: Path) -> pd.DataFrame | None:
    health_path = case_root.parent / "cfd_case_health.csv"
    if health_path.exists():
        return pd.read_csv(health_path)
    return None


def _manifest_row(row: dict[str, object], raw_dir: Path) -> dict[str, object]:
    return {
        "case_id": row["case_id"],
        "source_case_id": row["source_case_id"],
        "shape": row["shape"],
        "Re": row["Re"],
        "dy": row["dy"],
        "eps": row["eps"],
        "seed": row["seed"],
        "split": row["split"],
        "domain": row["domain"],
        "openfoam_time": row["openfoam_time"],
        "convective_tau": row["convective_tau"],
        "snapshot_index": row["snapshot_index"],
        "status": "success",
        "metadata_json": str(raw_dir / str(row["case_id"]) / "metadata.json"),
        "wake_field_npz": row["wake_field_npz"],
    }


def build_case_wake_field(
    *,
    case_dir: Path,
    cfg: dict[str, Any],
    run_raw_dir: Path,
    openfoam_bashrc: str,
    time_dir: Path | None = None,
    snapshot_index: int = 0,
) -> dict[str, object]:
    meta = read_json(case_dir / "case_metadata.json")
    case = meta["case"]
    geometry_meta = meta["geometry"]
    geom = cfd_geometry(cfg)
    vision_cfg = cfg.get("vision", {})
    field_size = int(vision_cfg.get("field_size", 128))
    scales = [str(scale) for scale in vision_cfg.get("scales", ["distD1.0_full"])]
    channels = [str(channel) for channel in vision_cfg.get("channels", SUPPORTED_CHANNELS)]
    if channels != list(SUPPORTED_CHANNELS):
        raise ValueError(
            "CFD wake builder currently writes ux/uy/speed/vorticity channels together"
        )

    files = _sample_files(case_dir) if time_dir is None else []
    if files:
        x, y, ux, uy = _load_sample_points(files[-1])
        source_file = str(files[-1])
        openfoam_time = None
        convective_tau = None
        output_case_id = str(case["case_id"])
    else:
        x, y, ux, uy = _load_cell_centre_points(case_dir, openfoam_bashrc, time_dir=time_dir)
        velocity_dir = _latest_time_dir(case_dir) if time_dir is None else time_dir
        openfoam_time = _parse_time_dir(velocity_dir)
        convective_tau = (
            openfoam_time
            * float(case["inlet_u_m_s"])
            / float(geometry_meta["equivalent_diameter_m"])
        )
        source_file = str(velocity_dir / "U")
        output_case_id = str(case["case_id"])
        if time_dir is not None:
            output_case_id = f"{output_case_id}_{_time_tag(openfoam_time)}"
    x_start = float(case["obstacle_x_m"]) + 0.5 * float(geometry_meta["equivalent_diameter_m"])
    x_end = geom.tank_length_m
    y_min = 0.0
    y_max = geom.channel_height_m
    x_coords = np.linspace(x_start, x_end, field_size, dtype=np.float64)
    y_coords = np.linspace(y_min, y_max, field_size, dtype=np.float64)
    ux_grid, uy_grid = _interpolate_field(
        x=x,
        y=y,
        ux=ux,
        uy=uy,
        x_coords=x_coords,
        y_coords=y_coords,
    )
    field_raw = _derive_field(ux_grid, uy_grid, x_coords=x_coords, y_coords=y_coords)
    field_norm, channel_mean, channel_std = normalize_field(field_raw)
    crop_boxes = build_crop_boxes(
        field_raw[3],
        scales=scales,
        physical_h=geom.channel_height_m,
        physical_obstacle_d=float(geometry_meta["equivalent_diameter_m"]),
        canvas_x_start=x_start,
        canvas_x_end=x_end,
        canvas_y_min=y_min,
        canvas_y_max=y_max,
    )
    crops = np.stack(
        [resize_crop(field_norm, crop_boxes[scale], output_size=field_size) for scale in scales],
        axis=0,
    )

    raw_case_dir = run_raw_dir / output_case_id
    raw_case_dir.mkdir(parents=True, exist_ok=True)
    wake_field_path = raw_case_dir / WAKE_FIELD_FILENAME
    np.savez_compressed(
        wake_field_path,
        field_raw=field_raw,
        field_norm=field_norm,
        crops=crops,
        scales=np.asarray(scales),
        crop_boxes=np.asarray([crop_boxes[scale] for scale in scales], dtype=np.float32),
        channel_names=np.asarray(channels),
        channel_mean=channel_mean,
        channel_std=channel_std,
        source_frames=np.asarray(0, dtype=np.int32),
        flow_pair_count=np.asarray(0, dtype=np.int32),
    )

    metadata = {
        "case_id": output_case_id,
        "source_case_id": case["case_id"],
        "backend": "openfoam",
        "domain": "cfd",
        "shape": case["shape"],
        "Re": int(case["re_value"]),
        "dy": float(case["dy_m"]),
        "eps": 0.0,
        "seed": 0,
        "split": case["split"],
        "openfoam_time": openfoam_time,
        "convective_tau": convective_tau,
        "snapshot_index": int(snapshot_index),
        "geometry": {
            "H": geom.channel_height_m,
            "d": float(geometry_meta["equivalent_diameter_m"]),
            "equivalent_diameter": float(geometry_meta["equivalent_diameter_m"]),
            "beta_area": float(geometry_meta["obstacle_area_m2"]) / float(geom.channel_height_m**2),
            "x0": float(case["obstacle_x_m"]),
            "y0_nominal": geom.obstacle_y_m,
            "y0_actual": float(case["obstacle_y_m"]),
            "L_total": geom.tank_length_m,
        },
        "wake_roi": {
            "x_min": x_start,
            "x_max": x_end,
            "y_min": y_min,
            "y_max": y_max,
            "width": x_end - x_start,
            "height": y_max - y_min,
            "pixels_x": field_size,
            "pixels_y": field_size,
        },
        "field_channels": channels,
        "crop_boxes": crop_boxes,
        "field_canvas": {
            "x_start": x_start,
            "x_end": x_end,
            "y_min": y_min,
            "y_max": y_max,
        },
        "files": {
            "wake_field_npz": WAKE_FIELD_FILENAME,
            "source_openfoam_case": str(case_dir),
            "source_field_file": source_file,
        },
    }
    write_json(raw_case_dir / "metadata.json", metadata)

    row: dict[str, object] = {
        "case_id": output_case_id,
        "source_case_id": str(case["case_id"]),
        "shape": str(case["shape"]),
        "Re": int(case["re_value"]),
        "dy": float(case["dy_m"]),
        "eps": 0.0,
        "seed": 0,
        "split": str(case["split"]),
        "domain": "cfd",
        "openfoam_time": openfoam_time,
        "convective_tau": convective_tau,
        "snapshot_index": int(snapshot_index),
        "wake_field_npz": str(wake_field_path),
        "wake_frames": 0,
        "field_size": field_size,
        "channels": "|".join(channels),
        "scales": "|".join(scales),
        "flow_pair_count": 0,
        "canvas_x_start": x_start,
        "canvas_x_end": x_end,
        "canvas_y_min": y_min,
        "canvas_y_max": y_max,
    }
    for scale, box in crop_boxes.items():
        row[f"{scale}_box"] = "|".join(f"{value:.6f}" for value in box)
    return row


def _build_case_outputs(
    *,
    case_dir: Path,
    cfg: dict[str, Any],
    raw_dir: Path,
    openfoam_bashrc: str,
    time_mode: str,
    time_min: float,
    tau_min: float | None,
    tau_max: float | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    rows = []
    manifest_rows = []
    skipped = 0
    meta = read_json(case_dir / "case_metadata.json")
    case_id = str(meta["case"]["case_id"])
    try:
        time_dirs = _selected_time_dirs(
            case_dir,
            time_mode=time_mode,
            time_min=time_min,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        for snapshot_index, time_dir in enumerate(time_dirs):
            row = build_case_wake_field(
                case_dir=case_dir,
                cfg=cfg,
                run_raw_dir=raw_dir,
                openfoam_bashrc=openfoam_bashrc,
                time_dir=time_dir,
                snapshot_index=snapshot_index,
            )
            rows.append(row)
            manifest_rows.append(_manifest_row(row, raw_dir))
    except Exception as exc:  # pylint: disable=broad-except
        skipped += 1
        manifest_rows.append(
            {
                "case_id": case_id,
                "status": "failed",
                "error": str(exc),
            }
        )
    return rows, manifest_rows, skipped


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    case_root = Path(args.case_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    raw_dir = run_dir / "data" / "raw"
    wake_dir = run_dir / "data" / "wake_fields"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wake_dir.mkdir(parents=True, exist_ok=True)

    health_df = _load_health(case_root)
    rows = []
    manifest_rows = []
    skipped = 0
    runnable_cases = []
    for case_dir in _case_dirs(case_root):
        meta = read_json(case_dir / "case_metadata.json")
        case_id = str(meta["case"]["case_id"])
        status = _status_for_case(case_id, health_df)
        if status not in {"success", "unknown"}:
            skipped += 1
            continue
        runnable_cases.append(case_dir)

    workers = max(1, int(args.workers))
    if workers == 1:
        for case_dir in runnable_cases:
            case_rows, case_manifest_rows, case_skipped = _build_case_outputs(
                case_dir=case_dir,
                cfg=cfg,
                raw_dir=raw_dir,
                openfoam_bashrc=args.openfoam_bashrc,
                time_mode=args.time_mode,
                time_min=float(args.time_min),
                tau_min=args.tau_min,
                tau_max=args.tau_max,
            )
            rows.extend(case_rows)
            manifest_rows.extend(case_manifest_rows)
            skipped += case_skipped
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _build_case_outputs,
                    case_dir=case_dir,
                    cfg=cfg,
                    raw_dir=raw_dir,
                    openfoam_bashrc=args.openfoam_bashrc,
                    time_mode=args.time_mode,
                    time_min=float(args.time_min),
                    tau_min=args.tau_min,
                    tau_max=args.tau_max,
                )
                for case_dir in runnable_cases
            ]
            for future in concurrent.futures.as_completed(futures):
                case_rows, case_manifest_rows, case_skipped = future.result()
                rows.extend(case_rows)
                manifest_rows.extend(case_manifest_rows)
                skipped += case_skipped

    if not rows:
        raise RuntimeError("No CFD wake fields were built")

    pd.DataFrame(rows).sort_values("case_id").to_csv(wake_dir / "index.csv", index=False)
    pd.DataFrame(manifest_rows).sort_values("case_id").to_csv(raw_dir / "manifest.csv", index=False)
    pd.DataFrame(manifest_rows).sort_values("case_id").to_csv(raw_dir / "index.csv", index=False)
    summary = {
        "rows": len(rows),
        "skipped": skipped,
        "index_csv": str(wake_dir / "index.csv"),
        "time_mode": args.time_mode,
        "time_min": float(args.time_min),
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
    }
    (wake_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(rows)} CFD wake fields; skipped={skipped}; index={wake_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
