from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import ConvexHull

from sim.config import load_config, resolve_from_root

WATER_NU_M2_S = 1.0e-6


@dataclass(frozen=True)
class CfdGeometry:
    tank_length_m: float
    channel_height_m: float
    tank_width_m: float
    obstacle_area_m2: float
    equivalent_diameter_m: float
    obstacle_x_m: float
    obstacle_y_m: float
    nu_m2_s: float


@dataclass(frozen=True)
class CfdCase:
    case_id: str
    shape: str
    re_value: int
    dy_m: float
    split: str
    inlet_u_m_s: float
    obstacle_x_m: float
    obstacle_y_m: float
    stl_path: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def read_config(path: str | Path) -> dict[str, Any]:
    return load_config(str(path))


def cfd_geometry(cfg: dict[str, Any]) -> CfdGeometry:
    cfd_cfg = cfg.get("cfd", {})
    sim_cfg = cfg.get("simulation", {})
    tank_cfg = cfd_cfg.get("tank", {})
    obstacle_cfg = cfd_cfg.get("obstacle", {})

    tank_length = float(
        tank_cfg.get("length_m", sim_cfg.get("L_in", 0.25) + sim_cfg.get("L_out", 0.6))
    )
    channel_height = float(tank_cfg.get("depth_m", sim_cfg.get("H", 0.45)))
    tank_width = float(tank_cfg.get("width_m", 0.40))
    obstacle_area = float(obstacle_cfg.get("area_m2", 0.005))
    equivalent_diameter = float(
        obstacle_cfg.get("equivalent_diameter_m", math.sqrt(4.0 * obstacle_area / math.pi))
    )
    obstacle_x = float(obstacle_cfg.get("center_x_m", sim_cfg.get("x0", 0.25)))
    obstacle_y = float(obstacle_cfg.get("center_y_m", sim_cfg.get("y0", 0.5 * channel_height)))
    nu = float(cfd_cfg.get("fluid", {}).get("nu_m2_s", WATER_NU_M2_S))

    return CfdGeometry(
        tank_length_m=tank_length,
        channel_height_m=channel_height,
        tank_width_m=tank_width,
        obstacle_area_m2=obstacle_area,
        equivalent_diameter_m=equivalent_diameter,
        obstacle_x_m=obstacle_x,
        obstacle_y_m=obstacle_y,
        nu_m2_s=nu,
    )


def inlet_u_from_re(re_value: int, geometry: CfdGeometry) -> float:
    return float(re_value) * geometry.nu_m2_s / geometry.equivalent_diameter_m


def dy_tag(dy_m: float) -> str:
    if abs(dy_m) < 1.0e-12:
        return "dy0"
    sign = "p" if dy_m > 0.0 else "m"
    return f"dy{sign}{abs(dy_m) * 1000.0:.0f}mm"


def shape_from_stl(path: Path) -> str:
    return path.stem.split("_", 1)[0].lower()


def resolve_stl_map(stl_dir: str | Path, shapes: list[str]) -> dict[str, Path]:
    root = resolve_from_root(stl_dir)
    if not root.exists():
        raise FileNotFoundError(f"STL directory not found: {root}")
    found = {shape_from_stl(path): path for path in root.glob("*.STL")}
    found.update({shape_from_stl(path): path for path in root.glob("*.stl")})
    missing = [shape for shape in shapes if shape not in found]
    if missing:
        raise FileNotFoundError(f"Missing STL files for shapes: {missing} in {root}")
    return {shape: found[shape] for shape in shapes}


def case_matrix(cfg: dict[str, Any], stl_dir: str | Path) -> list[CfdCase]:
    cfd_cfg = cfg.get("cfd", {})
    matrix_cfg = cfd_cfg.get("case_matrix", {})
    sim_cfg = cfg.get("simulation", {})
    geometry = cfd_geometry(cfg)

    shapes = [str(shape) for shape in matrix_cfg.get("shapes", sim_cfg.get("shapes", []))]
    re_values = [int(value) for value in matrix_cfg.get("re_values", sim_cfg.get("re_values", []))]
    dy_values = [float(value) for value in matrix_cfg.get("dy_values_m", [0.0])]
    test_dy_values = {
        round(float(value), 12) for value in matrix_cfg.get("test_dy_values_m", [0.0])
    }
    stl_map = resolve_stl_map(stl_dir, shapes)

    cases: list[CfdCase] = []
    for shape in shapes:
        for re_value in re_values:
            for dy_m in dy_values:
                split = "test" if round(dy_m, 12) in test_dy_values else "train"
                case_id = f"{shape}_Re{re_value}_{dy_tag(dy_m)}"
                cases.append(
                    CfdCase(
                        case_id=case_id,
                        shape=shape,
                        re_value=re_value,
                        dy_m=dy_m,
                        split=split,
                        inlet_u_m_s=inlet_u_from_re(re_value, geometry),
                        obstacle_x_m=geometry.obstacle_x_m,
                        obstacle_y_m=geometry.obstacle_y_m + dy_m,
                        stl_path=str(stl_map[shape]),
                    )
                )
    return cases


def read_stl_vertices(path: str | Path) -> np.ndarray:
    stl_path = Path(path)
    data = stl_path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    head = data[:256].decode("utf-8", errors="ignore").lower()
    if head.lstrip().startswith("solid") and b"vertex" in data[:4096].lower():
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    elif len(data) >= 84:
        n_triangles = struct.unpack("<I", data[80:84])[0]
        offset = 84
        for _ in range(n_triangles):
            if offset + 50 > len(data):
                break
            values = struct.unpack("<12fH", data[offset : offset + 50])
            vertices.extend([values[3:6], values[6:9], values[9:12]])
            offset += 50

    if not vertices:
        raise ValueError(f"No STL vertices parsed from {stl_path}")
    return np.asarray(vertices, dtype=np.float64)


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def projected_outline_m(stl_path: str | Path, center_x_m: float, center_y_m: float) -> np.ndarray:
    vertices = read_stl_vertices(stl_path)
    xy_m = vertices[:, :2] * 0.001
    unique_xy = np.unique(np.round(xy_m, decimals=9), axis=0)
    if unique_xy.shape[0] < 3:
        raise ValueError(f"Need at least 3 unique projected STL vertices: {stl_path}")

    hull = ConvexHull(unique_xy)
    outline = unique_xy[hull.vertices]
    if _signed_area(outline) > 0.0:
        outline = outline[::-1]

    center = np.array(
        [
            (outline[:, 0].min() + outline[:, 0].max()) * 0.5,
            (outline[:, 1].min() + outline[:, 1].max()) * 0.5,
        ]
    )
    shifted = outline - center + np.array([center_x_m, center_y_m])
    return cast(np.ndarray, shifted.astype(np.float64))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def parse_openfoam_time(path: Path) -> float:
    try:
        return float(path.name)
    except ValueError:
        return -1.0


def latest_time_dir(case_dir: Path) -> Path | None:
    candidates = [
        path
        for path in case_dir.iterdir()
        if path.is_dir() and path.name not in {"0", "constant", "system", "logs"}
    ]
    if not candidates:
        return None
    return max(candidates, key=parse_openfoam_time)


def parse_float_line(line: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
