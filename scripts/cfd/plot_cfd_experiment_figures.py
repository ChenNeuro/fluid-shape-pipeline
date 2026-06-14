from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scripts.cfd.common import (  # noqa: E402
    cfd_geometry,
    projected_outline_m,
    read_config,
    resolve_stl_map,
)

DEFAULT_INDEX = Path(
    "/home/chenyihao/fluid_runs/cfd_tank_stable175_tau6/data/wake_fields/index.csv"
)
DEFAULT_OUTPUT_DIR = Path("docs/figures/cfd_experiment")
DEFAULT_CONFIG = Path("configs/cfd_tank_stable175.yaml")
DEFAULT_STL_DIR = Path("HURRY/solidworks_model_STL")
SHAPES = ["circle", "triangle", "airfoil", "diamond", "bar"]
CHANNELS = ["ux", "uy", "speed", "vorticity"]


def _font_properties() -> FontProperties | None:
    candidates = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


def _set_plot_style() -> FontProperties | None:
    font = _font_properties()
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.grid": False,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    )
    return font


def _title(ax: plt.Axes, text: str, font: FontProperties | None) -> None:
    ax.set_title(text, fontproperties=font if font else None)


def _label_x(ax: plt.Axes, text: str, font: FontProperties | None) -> None:
    ax.set_xlabel(text, fontproperties=font if font else None)


def _label_y(ax: plt.Axes, text: str, font: FontProperties | None) -> None:
    ax.set_ylabel(text, fontproperties=font if font else None)


def _load_wake(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _channel(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    idx = CHANNELS.index(name)
    return np.asarray(data["field_raw"][idx], dtype=float)


def _extent(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["canvas_x_start"]),
        float(row["canvas_x_end"]),
        float(row["canvas_y_min"]),
        float(row["canvas_y_max"]),
    )


def _inlet_u(re_value: float, diameter_m: float, nu_m2_s: float) -> float:
    return float(re_value) * nu_m2_s / diameter_m


def _latest_rows(
    df: pd.DataFrame,
    *,
    shapes: Iterable[str],
    re_value: int,
    dy_m: float,
) -> pd.DataFrame:
    rows = []
    for shape in shapes:
        sub = df[
            (df["shape"] == shape)
            & (df["Re"] == re_value)
            & (np.isclose(df["dy"], dy_m, atol=1.0e-9))
        ].sort_values("convective_tau")
        if sub.empty:
            raise ValueError(f"No row for shape={shape}, Re={re_value}, dy={dy_m}")
        rows.append(sub.iloc[-1])
    return pd.DataFrame(rows)


def _nearest_tau_rows(
    df: pd.DataFrame,
    *,
    shape: str,
    re_value: int,
    dy_m: float,
    taus: Iterable[float],
) -> pd.DataFrame:
    rows = []
    sub = df[
        (df["shape"] == shape) & (df["Re"] == re_value) & (np.isclose(df["dy"], dy_m, atol=1.0e-9))
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for shape={shape}, Re={re_value}, dy={dy_m}")
    for tau in taus:
        idx = (sub["convective_tau"] - tau).abs().idxmin()
        rows.append(sub.loc[idx])
    return pd.DataFrame(rows)


def _latest_re_rows(
    df: pd.DataFrame,
    *,
    shape: str,
    re_values: Iterable[int],
    dy_m: float,
) -> pd.DataFrame:
    rows = []
    for re_value in re_values:
        sub = df[
            (df["shape"] == shape)
            & (df["Re"] == re_value)
            & (np.isclose(df["dy"], dy_m, atol=1.0e-9))
        ].sort_values("convective_tau")
        if sub.empty:
            raise ValueError(f"No row for shape={shape}, Re={re_value}, dy={dy_m}")
        rows.append(sub.iloc[-1])
    return pd.DataFrame(rows)


def _add_colorbar(fig: plt.Figure, image, ax: plt.Axes, label: str) -> None:
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.set_ylabel(label, rotation=90)


def _plot_shape_outlines(
    output_dir: Path,
    stl_dir: Path,
    shapes: list[str],
    font: FontProperties | None,
) -> Path:
    stl_map = resolve_stl_map(stl_dir, shapes)
    fig, axes = plt.subplots(1, len(shapes), figsize=(13, 3.0), constrained_layout=True)
    for ax, shape in zip(axes, shapes):
        outline = projected_outline_m(stl_map[shape], 0.0, 0.0) * 1000.0
        outline = np.vstack([outline, outline[0]])
        ax.fill(outline[:, 0], outline[:, 1], color="#5b8fd8", alpha=0.35)
        ax.plot(outline[:, 0], outline[:, 1], color="#174a8b", linewidth=1.4)
        width = outline[:, 0].max() - outline[:, 0].min()
        height = outline[:, 1].max() - outline[:, 1].min()
        _title(ax, f"{shape}\n{width:.1f} x {height:.1f} mm", font)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0.0, color="0.75", linewidth=0.6)
        ax.axvline(0.0, color="0.75", linewidth=0.6)
    fig.suptitle("Equal-area STL projected outlines, A = 5000 mm^2", y=1.04)
    path = output_dir / "01_equal_area_stl_outlines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_tank_roi(
    output_dir: Path,
    cfg: dict,
    index_rows: pd.DataFrame,
    stl_dir: Path,
    font: FontProperties | None,
) -> Path:
    geometry = cfd_geometry(cfg)
    stl_map = resolve_stl_map(stl_dir, ["circle"])
    circle_outline = projected_outline_m(
        stl_map["circle"], geometry.obstacle_x_m, geometry.obstacle_y_m
    )
    row = index_rows[index_rows["shape"] == "circle"].iloc[0]
    x0, x1, y0, y1 = _extent(row)
    diameter = geometry.equivalent_diameter_m

    fig, ax = plt.subplots(figsize=(11.5, 4.2), constrained_layout=True)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            geometry.tank_length_m,
            geometry.channel_height_m,
            facecolor="#edf5ff",
            edgecolor="#234",
            linewidth=1.2,
        )
    )
    ax.fill(circle_outline[:, 0], circle_outline[:, 1], color="#444", alpha=0.8)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="#f7b267",
            alpha=0.18,
            edgecolor="#d47700",
            linewidth=1.2,
        )
    )
    colors = ["#0072b2", "#009e73", "#d55e00"]
    label_y = {
        1.0: geometry.channel_height_m + 0.048,
        2.0: geometry.channel_height_m + 0.026,
        4.0: geometry.channel_height_m + 0.048,
    }
    for multiple, color in zip([1.0, 2.0, 4.0], colors):
        x_start = x0 + multiple * diameter
        ax.axvline(x_start, color=color, linewidth=1.4, linestyle="--")
        ax.text(
            x_start + 0.004,
            label_y[multiple],
            f"{multiple:g}D crop",
            color=color,
            fontsize=8,
        )
    ax.arrow(
        0.045,
        geometry.channel_height_m + 0.026,
        0.13,
        0.0,
        width=0.003,
        head_width=0.016,
        head_length=0.018,
        color="#1b4d89",
        length_includes_head=True,
    )
    ax.text(0.045, geometry.channel_height_m + 0.045, "inlet flow", color="#1b4d89")
    ax.text(
        x0 + 0.025,
        0.022,
        "Wake ROI for CFD/PIV comparison",
        color="#9a4f00",
        fontsize=9,
    )
    ax.text(
        geometry.obstacle_x_m - 0.038,
        geometry.obstacle_y_m + diameter * 0.65,
        "obstacle\ncenter x=0.25 m",
        fontsize=8,
        ha="center",
    )
    ax.set_xlim(-0.015, geometry.tank_length_m + 0.015)
    ax.set_ylim(-0.03, geometry.channel_height_m + 0.08)
    ax.set_aspect("equal", adjustable="box")
    _label_x(ax, "x, streamwise direction (m)", font)
    _label_y(ax, "y, tank depth/channel height (m)", font)
    _title(
        ax,
        "Tank/experiment alignment: 0.85 m x 0.45 m 2D channel, width ignored in CFD",
        font,
    )
    path = output_dir / "02_tank_roi_alignment.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_shape_wakes(
    output_dir: Path,
    rows: pd.DataFrame,
    cfg: dict,
    font: FontProperties | None,
) -> Path:
    geometry = cfd_geometry(cfg)
    loaded = [(row, _load_wake(row["wake_field_npz"])) for _, row in rows.iterrows()]
    vort_values = []
    for row, data in loaded:
        u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
        vort_values.append(_channel(data, "vorticity") * geometry.equivalent_diameter_m / u_in)
    vort_lim = float(np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in vort_values])), 98))
    vort_lim = max(vort_lim, 1.0)

    fig, axes = plt.subplots(
        len(loaded),
        2,
        figsize=(11.5, 12.0),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for r, (row, data) in enumerate(loaded):
        extent = _extent(row)
        u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
        speed_ratio = _channel(data, "speed") / u_in
        vort_nd = _channel(data, "vorticity") * geometry.equivalent_diameter_m / u_in

        im0 = axes[r, 0].imshow(
            speed_ratio,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.35,
            aspect="auto",
        )
        im1 = axes[r, 1].imshow(
            vort_nd,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            vmin=-vort_lim,
            vmax=vort_lim,
            aspect="auto",
        )
        for ax in axes[r]:
            ax.set_ylim(0.0, geometry.channel_height_m)
            ax.set_xlim(extent[0], extent[1])
            ax.tick_params(labelsize=7)
        axes[r, 0].set_ylabel(f"{row['shape']}\ny (m)")
        _title(
            axes[r, 0],
            f"{row['shape']} speed/U, Re={int(row['Re'])}, tau={row['convective_tau']:.1f}",
            font,
        )
        _title(axes[r, 1], f"{row['shape']} vorticity*D/U", font)
        if r == len(loaded) - 1:
            _label_x(axes[r, 0], "x (m)", font)
            _label_x(axes[r, 1], "x (m)", font)
        _add_colorbar(fig, im0, axes[r, 0], "speed/U")
        _add_colorbar(fig, im1, axes[r, 1], "omega D/U")

    fig.suptitle("Stable CFD wake comparison at dy=0, latest tau snapshot", y=1.01)
    path = output_dir / "03_shape_wakes_re800_dy0_latest_tau.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_re_sweep(
    output_dir: Path,
    rows: pd.DataFrame,
    cfg: dict,
    font: FontProperties | None,
) -> Path:
    geometry = cfd_geometry(cfg)
    loaded = [(row, _load_wake(row["wake_field_npz"])) for _, row in rows.iterrows()]
    values = []
    for row, data in loaded:
        u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
        values.append(_channel(data, "vorticity") * geometry.equivalent_diameter_m / u_in)
    limit = float(np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in values])), 98))
    limit = max(limit, 1.0)

    fig, axes = plt.subplots(1, len(loaded), figsize=(15, 3.5), constrained_layout=True)
    for ax, (row, data) in zip(axes, loaded):
        extent = _extent(row)
        u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
        vort_nd = _channel(data, "vorticity") * geometry.equivalent_diameter_m / u_in
        im = ax.imshow(
            vort_nd,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(0.0, geometry.channel_height_m)
        _title(
            ax,
            f"Re={int(row['Re'])}\nU={u_in*1000:.1f} mm/s, tau={row['convective_tau']:.0f}",
            font,
        )
        _label_x(ax, "x (m)", font)
    _label_y(axes[0], "y (m)", font)
    _add_colorbar(fig, im, axes[-1], "omega D/U")
    fig.suptitle("Circle wake sensitivity to Reynolds number, dy=0", y=1.08)
    path = output_dir / "04_circle_re_sweep_latest_tau.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_crop_alignment(
    output_dir: Path,
    row: pd.Series,
    cfg: dict,
    font: FontProperties | None,
) -> Path:
    geometry = cfd_geometry(cfg)
    data = _load_wake(row["wake_field_npz"])
    extent = _extent(row)
    u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
    speed_ratio = _channel(data, "speed") / u_in

    fig, ax = plt.subplots(figsize=(11.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        speed_ratio,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.35,
        aspect="auto",
    )
    colors = ["#0072b2", "#009e73", "#d55e00"]
    for multiple, color in zip([1.0, 2.0, 4.0], colors):
        x_start = extent[0] + multiple * geometry.equivalent_diameter_m
        ax.axvline(x_start, color=color, linewidth=2.0, linestyle="--")
        ax.text(x_start + 0.006, extent[3] - 0.035, f"{multiple:g}D", color=color)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _title(
        ax,
        "PIV/CFD crop alignment: distD1.0, distD2.0, distD4.0 start lines",
        font,
    )
    _label_x(ax, "x (m), streamwise", font)
    _label_y(ax, "y (m), full depth", font)
    _add_colorbar(fig, im, ax, "speed/U")
    path = output_dir / "05_circle_crop_alignment_re800.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_time_evolution(
    output_dir: Path,
    rows: pd.DataFrame,
    cfg: dict,
    font: FontProperties | None,
) -> Path:
    geometry = cfd_geometry(cfg)
    fig, axes = plt.subplots(1, len(rows), figsize=(14, 3.4), constrained_layout=True)
    for ax, (_, row) in zip(axes, rows.iterrows()):
        data = _load_wake(row["wake_field_npz"])
        extent = _extent(row)
        u_in = _inlet_u(row["Re"], geometry.equivalent_diameter_m, geometry.nu_m2_s)
        speed_ratio = _channel(data, "speed") / u_in
        im = ax.imshow(
            speed_ratio,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.35,
            aspect="auto",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        _title(ax, f"tau={row['convective_tau']:.0f}\nt={row['openfoam_time']:.1f}s", font)
        _label_x(ax, "x (m)", font)
    _label_y(axes[0], "y (m)", font)
    _add_colorbar(fig, im, axes[-1], "speed/U")
    fig.suptitle("Circle Re=800 wake development after tau>=6", y=1.08)
    path = output_dir / "06_circle_time_evolution_re800.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stl-dir", type=Path, default=DEFAULT_STL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--re", type=int, default=800)
    parser.add_argument("--dy", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _set_plot_style()
    cfg = read_config(args.config)
    df = pd.read_csv(args.index)

    selected_rows = _latest_rows(df, shapes=SHAPES, re_value=args.re, dy_m=args.dy)
    circle_row = selected_rows[selected_rows["shape"] == "circle"].iloc[0]
    re_rows = _latest_re_rows(df, shape="circle", re_values=[300, 500, 800, 1100, 1500], dy_m=0.0)
    tau_rows = _nearest_tau_rows(df, shape="circle", re_value=800, dy_m=0.0, taus=[7, 8, 9, 10])

    outputs = [
        _plot_shape_outlines(output_dir, args.stl_dir, SHAPES, font),
        _plot_tank_roi(output_dir, cfg, selected_rows, args.stl_dir, font),
        _plot_shape_wakes(output_dir, selected_rows, cfg, font),
        _plot_re_sweep(output_dir, re_rows, cfg, font),
        _plot_crop_alignment(output_dir, circle_row, cfg, font),
        _plot_time_evolution(output_dir, tau_rows, cfg, font),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
