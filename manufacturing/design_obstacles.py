from __future__ import annotations

"""Obstacle dimension calculator for equal-area wake experiments.

The experiment uses an equal in-plane obstacle area for each shape:

    area_ratio = A / H^2

where H is the channel height and A is the obstacle area in the laser-sheet
plane.  This matches the simulation masks, where all dimensions scale with H.

For flow similarity, use the equivalent circular diameter

    D_eq = sqrt(4 A / pi)

and Reynolds number

    Re = U * D_eq / nu

where U is the characteristic inlet velocity and nu is the kinematic viscosity.
"""

import argparse
import math
from collections.abc import Iterable


def _naca_integral(thickness_ratio: float = 0.14) -> float:
    """Return the NACA 00xx area coefficient A / chord^2."""
    integral = (
        0.2969 * 2.0 / 3.0
        - 0.1260 * 0.5
        - 0.3516 * 1.0 / 3.0
        + 0.2843 * 0.25
        - 0.1015 * 0.2
    )
    return 2.0 * 5.0 * thickness_ratio * integral


def design_obstacles(
    H: float,
    blockage_ratio: float = 0.04,
    airfoil_thickness: float = 0.14,
    nu: float | None = None,
    velocity: float | None = None,
    target_re: float | None = None,
) -> dict:
    """Compute equal-area obstacle dimensions.

    Args:
        H: channel height in meters.
        blockage_ratio: in-plane area ratio A/H^2.
        airfoil_thickness: NACA symmetric airfoil thickness ratio.
        nu: fluid kinematic viscosity in m^2/s.
        velocity: characteristic inlet velocity in m/s.
        target_re: optional target Reynolds number based on D_eq.

    Returns:
        Dictionary with dimensions in meters and optional flow quantities.
    """
    if H <= 0:
        raise ValueError("H must be positive")
    if blockage_ratio <= 0:
        raise ValueError("blockage_ratio must be positive")

    area = blockage_ratio * H**2
    d_eq = math.sqrt(4.0 * area / math.pi)

    circle = {"diameter": d_eq, "radius": 0.5 * d_eq}

    tri_side = math.sqrt(4.0 * area / math.sqrt(3.0))
    tri_height = math.sqrt(3.0) / 2.0 * tri_side
    triangle = {
        "side": tri_side,
        "height": tri_height,
        "apex": (-2.0 / 3.0 * tri_height, 0.0),
        "base_left": (1.0 / 3.0 * tri_height, -tri_side / 2.0),
        "base_right": (1.0 / 3.0 * tri_height, tri_side / 2.0),
    }

    area_coeff = _naca_integral(airfoil_thickness)
    chord = math.sqrt(area / area_coeff)
    airfoil = {
        "chord": chord,
        "max_thickness": chord * airfoil_thickness,
        "leading_edge": (-chord / 2.0, 0.0),
        "trailing_edge": (chord / 2.0, 0.0),
    }

    diamond_aspect = 1.474
    diamond_width = math.sqrt(2.0 * area * diamond_aspect)
    diamond_height = diamond_width / diamond_aspect
    diamond = {
        "width_diag": diamond_width,
        "height_diag": diamond_height,
        "edge": 0.5 * math.sqrt(diamond_width**2 + diamond_height**2),
        "vertices": [
            (-diamond_width / 2.0, 0.0),
            (0.0, diamond_height / 2.0),
            (diamond_width / 2.0, 0.0),
            (0.0, -diamond_height / 2.0),
        ],
    }

    bar_aspect = 4.167
    bar_height = math.sqrt(area / bar_aspect)
    bar_width = bar_aspect * bar_height
    bar = {
        "width": bar_width,
        "height": bar_height,
        "aspect": bar_aspect,
        "vertices": [
            (-bar_width / 2.0, -bar_height / 2.0),
            (bar_width / 2.0, -bar_height / 2.0),
            (bar_width / 2.0, bar_height / 2.0),
            (-bar_width / 2.0, bar_height / 2.0),
        ],
    }

    flow = {
        "nu": nu,
        "velocity": velocity,
        "target_re": target_re,
        "re_at_velocity": None,
        "velocity_for_target_re": None,
    }
    if nu is not None and nu <= 0:
        raise ValueError("nu must be positive when provided")
    if velocity is not None and velocity < 0:
        raise ValueError("velocity must be non-negative when provided")
    if target_re is not None and target_re <= 0:
        raise ValueError("target_re must be positive when provided")
    if nu is not None and velocity is not None:
        flow["re_at_velocity"] = velocity * d_eq / nu
    if nu is not None and target_re is not None:
        flow["velocity_for_target_re"] = target_re * nu / d_eq

    return {
        "channel_height": H,
        "area_ratio": blockage_ratio,
        "target_area": area,
        "D_eq": d_eq,
        "D_eq_over_H": d_eq / H,
        "circle": circle,
        "triangle": triangle,
        "airfoil": airfoil,
        "diamond": diamond,
        "bar": bar,
        "flow": flow,
    }


def _mm(value: float) -> str:
    return f"{value * 1000.0:.1f} mm"


def _mm2(value: float) -> str:
    return f"{value * 1e6:.0f} mm^2"


def _parse_betas(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def print_design(design: dict) -> None:
    area = design["target_area"]
    print(f"# Channel height H = {design['channel_height']:.4f} m")
    print(
        "# Area ratio beta_area = A/H^2 = "
        f"{design['area_ratio']:.4f} ({design['area_ratio'] * 100:.1f}%)"
    )
    print(f"# Target area A = {area:.6f} m^2 = {_mm2(area)}")
    print(f"# Equivalent diameter D_eq = {_mm(design['D_eq'])}")
    print(f"# D_eq/H = {design['D_eq_over_H']:.4f}")

    flow = design["flow"]
    if flow["nu"] is not None:
        print(f"# Fluid kinematic viscosity nu = {flow['nu']:.3e} m^2/s")
    if flow["re_at_velocity"] is not None:
        print(
            f"# Re at U={flow['velocity']:.4f} m/s: "
            f"{flow['re_at_velocity']:.1f}"
        )
    if flow["velocity_for_target_re"] is not None:
        print(
            f"# U for Re={flow['target_re']:.1f}: "
            f"{flow['velocity_for_target_re']:.5f} m/s"
        )

    print()
    print("=" * 78)
    print(f"{'Shape':<12} {'Key dim 1':<22} {'Key dim 2':<22} {'Area':>14}")
    print("=" * 78)
    print(
        f"{'Circle':<12} {'D = ' + _mm(design['circle']['diameter']):<22} "
        f"{'--':<22} {_mm2(area):>14}"
    )
    print(
        f"{'Triangle':<12} {'a = ' + _mm(design['triangle']['side']):<22} "
        f"{'h = ' + _mm(design['triangle']['height']):<22} {_mm2(area):>14}"
    )
    print(
        f"{'Airfoil':<12} {'c = ' + _mm(design['airfoil']['chord']):<22} "
        f"{'t = ' + _mm(design['airfoil']['max_thickness']):<22} {_mm2(area):>14}"
    )
    print(
        f"{'Diamond':<12} {'w = ' + _mm(design['diamond']['width_diag']):<22} "
        f"{'h = ' + _mm(design['diamond']['height_diag']):<22} {_mm2(area):>14}"
    )
    print(
        f"{'Bar':<12} {'w = ' + _mm(design['bar']['width']):<22} "
        f"{'h = ' + _mm(design['bar']['height']):<22} {_mm2(area):>14}"
    )
    print("=" * 78)


def print_sweep(
    *,
    H: float,
    betas: Iterable[float],
    nu: float | None,
    velocity: float | None,
    target_re: float | None,
) -> None:
    header = [
        "beta_area",
        "D_eq/H",
        "A_mm2",
        "circle_D_mm",
        "airfoil_c_mm",
        "bar_w_mm",
        "bar_h_mm",
    ]
    if nu is not None and velocity is not None:
        header.append("Re_at_U")
    if nu is not None and target_re is not None:
        header.append("U_for_target_Re_mps")

    print(",".join(header))
    for beta in betas:
        design = design_obstacles(
            H=H,
            blockage_ratio=beta,
            nu=nu,
            velocity=velocity,
            target_re=target_re,
        )
        row: list[str] = [
            f"{beta:.4f}",
            f"{design['D_eq_over_H']:.4f}",
            f"{design['target_area'] * 1e6:.1f}",
            f"{design['circle']['diameter'] * 1000.0:.1f}",
            f"{design['airfoil']['chord'] * 1000.0:.1f}",
            f"{design['bar']['width'] * 1000.0:.1f}",
            f"{design['bar']['height'] * 1000.0:.1f}",
        ]
        flow = design["flow"]
        if flow["re_at_velocity"] is not None:
            row.append(f"{flow['re_at_velocity']:.1f}")
        if flow["velocity_for_target_re"] is not None:
            row.append(f"{flow['velocity_for_target_re']:.5f}")
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Design equal-area obstacles and flow-compatible beta sweeps"
    )
    parser.add_argument("--H", type=float, default=1.0, help="Channel height in meters")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Area ratio A/H^2, e.g. 0.04 for 4%%",
    )
    parser.add_argument(
        "--sweep-beta",
        default=None,
        help="Comma-separated beta_area values, e.g. 0.02,0.03,0.04,0.05",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=None,
        help="Fluid kinematic viscosity in m^2/s",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=None,
        help="Characteristic inlet velocity in m/s",
    )
    parser.add_argument(
        "--target-re",
        type=float,
        default=None,
        help="Target Reynolds number based on D_eq",
    )
    args = parser.parse_args()

    if args.sweep_beta:
        print_sweep(
            H=args.H,
            betas=_parse_betas(args.sweep_beta),
            nu=args.nu,
            velocity=args.velocity,
            target_re=args.target_re,
        )
        return

    print_design(
        design_obstacles(
            H=args.H,
            blockage_ratio=args.beta,
            nu=args.nu,
            velocity=args.velocity,
            target_re=args.target_re,
        )
    )


if __name__ == "__main__":
    main()
