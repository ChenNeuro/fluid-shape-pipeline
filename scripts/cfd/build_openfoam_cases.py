from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pandas as pd

from scripts.cfd.common import (
    CfdCase,
    case_matrix,
    cfd_geometry,
    projected_outline_m,
    read_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D OpenFOAM/Gmsh cases from STL outlines")
    parser.add_argument("--config", required=True, help="CFD YAML config")
    parser.add_argument("--stl-dir", required=True, help="Directory containing shape STL files")
    parser.add_argument(
        "--output-root", required=True, help="Output root, preferably under WSL /home"
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Only write case files; do not invoke gmsh/gmshToFoam/checkMesh",
    )
    parser.add_argument(
        "--openfoam-bashrc",
        default="/usr/share/openfoam/etc/bashrc",
        help="OpenFOAM bashrc to source before gmshToFoam/checkMesh",
    )
    return parser.parse_args()


def _foam_header(class_name: str, object_name: str) -> str:
    return textwrap.dedent(f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {class_name};
            object      {object_name};
        }}
        """)


def _write_control_dict(case_dir: Path, cfg: dict, case: CfdCase) -> None:
    cfd_cfg = cfg.get("cfd", {})
    time_cfg = cfd_cfg.get("time", {})
    dt = float(time_cfg.get("dt_s", 0.002))
    end_time = float(time_cfg.get("end_time_s", 2.0))
    write_interval = float(time_cfg.get("write_interval_s", 0.05))
    geom = cfd_geometry(cfg)
    if "convective_end_tau" in time_cfg:
        end_time = (
            float(time_cfg["convective_end_tau"])
            * geom.equivalent_diameter_m
            / float(case.inlet_u_m_s)
        )
    if "convective_write_tau" in time_cfg:
        write_interval = (
            float(time_cfg["convective_write_tau"])
            * geom.equivalent_diameter_m
            / (float(case.inlet_u_m_s))
        )
    max_co = float(time_cfg.get("max_co", 0.7))
    adjust_time_step = "yes" if bool(time_cfg.get("adjust_time_step", False)) else "no"
    max_delta_t = float(time_cfg.get("max_delta_t_s", dt))

    text = _foam_header("dictionary", "controlDict") + textwrap.dedent(f"""\

            application     pimpleFoam;
            startFrom       startTime;
            startTime       0;
            stopAt          endTime;
            endTime         {end_time:.8g};
            deltaT          {dt:.8g};
            writeControl    adjustableRunTime;
            writeInterval   {write_interval:.8g};
            purgeWrite      0;
            writeFormat     ascii;
            writePrecision  8;
            writeCompression off;
            timeFormat      general;
            timePrecision   6;
            runTimeModifiable true;
            adjustTimeStep  {adjust_time_step};
            maxCo           {max_co:.8g};
            maxDeltaT       {max_delta_t:.8g};

            functions
            {{
            }}

            // case_id {case.case_id}
            """)
    (case_dir / "system" / "controlDict").write_text(text, encoding="utf-8")


def _write_fv_files(case_dir: Path) -> None:
    fv_schemes = _foam_header("dictionary", "fvSchemes") + textwrap.dedent("""\

            ddtSchemes
            {
                default         Euler;
            }

            gradSchemes
            {
                default         Gauss linear;
            }

            divSchemes
            {
                default         none;
                div(phi,U)      Gauss linearUpwind grad(U);
                div((nuEff*dev2(T(grad(U))))) Gauss linear;
            }

            laplacianSchemes
            {
                default         Gauss linear corrected;
            }

            interpolationSchemes
            {
                default         linear;
            }

            snGradSchemes
            {
                default         corrected;
            }
            """)
    fv_solution = _foam_header("dictionary", "fvSolution") + textwrap.dedent("""\

            solvers
            {
                p
                {
                    solver          GAMG;
                    tolerance       1e-7;
                    relTol          0.05;
                    smoother        DICGaussSeidel;
                }

                U
                {
                    solver          smoothSolver;
                    smoother        symGaussSeidel;
                    tolerance       1e-7;
                    relTol          0.05;
                }

                pFinal
                {
                    $p;
                    relTol          0;
                }

                UFinal
                {
                    $U;
                    relTol          0;
                }
            }

            PIMPLE
            {
                nOuterCorrectors 1;
                nCorrectors      2;
                nNonOrthogonalCorrectors 0;
            }
            """)
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes, encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text(fv_solution, encoding="utf-8")


def _write_fields(case_dir: Path, case: CfdCase, nu_m2_s: float) -> None:
    u_text = _foam_header("volVectorField", "U") + textwrap.dedent(f"""\

            dimensions      [0 1 -1 0 0 0 0];
            internalField   uniform ({case.inlet_u_m_s:.10g} 0 0);

            boundaryField
            {{
                inlet
                {{
                    type fixedValue;
                    value uniform ({case.inlet_u_m_s:.10g} 0 0);
                }}
                outlet
                {{
                    type zeroGradient;
                }}
                walls
                {{
                    type noSlip;
                }}
                obstacle
                {{
                    type noSlip;
                }}
                frontAndBack
                {{
                    type empty;
                }}
            }}
            """)
    p_text = _foam_header("volScalarField", "p") + textwrap.dedent("""\

            dimensions      [0 2 -2 0 0 0 0];
            internalField   uniform 0;

            boundaryField
            {
                inlet
                {
                    type zeroGradient;
                }
                outlet
                {
                    type fixedValue;
                    value uniform 0;
                }
                walls
                {
                    type zeroGradient;
                }
                obstacle
                {
                    type zeroGradient;
                }
                frontAndBack
                {
                    type empty;
                }
            }
            """)
    transport = _foam_header("dictionary", "transportProperties") + textwrap.dedent(f"""\

            transportModel  Newtonian;
            nu              [0 2 -1 0 0 0 0] {nu_m2_s:.10g};
            """)
    turbulence = _foam_header("dictionary", "turbulenceProperties") + textwrap.dedent("""\

            simulationType laminar;
            """)
    (case_dir / "0" / "U").write_text(u_text, encoding="utf-8")
    (case_dir / "0" / "p").write_text(p_text, encoding="utf-8")
    (case_dir / "constant" / "transportProperties").write_text(transport, encoding="utf-8")
    (case_dir / "constant" / "turbulenceProperties").write_text(turbulence, encoding="utf-8")


def _write_change_dictionary(case_dir: Path) -> None:
    text = _foam_header("dictionary", "changeDictionaryDict") + textwrap.dedent("""\

            dictionaryReplacement
            {
                boundary
                {
                    inlet
                    {
                        type patch;
                    }
                    outlet
                    {
                        type patch;
                    }
                    walls
                    {
                        type wall;
                    }
                    obstacle
                    {
                        type wall;
                    }
                    frontAndBack
                    {
                        type empty;
                    }
                }
            }
            """)
    (case_dir / "system" / "changeDictionaryDict").write_text(text, encoding="utf-8")


def _write_sample_dict(case_dir: Path, cfg: dict) -> None:
    cfd_cfg = cfg.get("cfd", {})
    geom = cfd_geometry(cfg)
    z_mid = float(cfd_cfg.get("mesh", {}).get("thickness_m", 0.01)) * 0.5
    text = _foam_header("dictionary", "sampleDict") + textwrap.dedent(f"""\

            type sets;
            libs ("libsampling.so");
            interpolationScheme cellPoint;
            setFormat raw;
            surfaceFormat raw;
            fields (U);

            surfaces
            (
                wakePlane
                {{
                    type cuttingPlane;
                    planeType pointAndNormal;
                    pointAndNormalDict
                    {{
                        point (0 0 {z_mid:.8g});
                        normal (0 0 1);
                    }}
                    interpolate true;
                }}
            );

            // Sampling domain L={geom.tank_length_m:.8g}, H={geom.channel_height_m:.8g}
            """)
    (case_dir / "system" / "sampleDict").write_text(text, encoding="utf-8")


def _replace_boundary_patch_type(text: str, patch_name: str, patch_type: str) -> str:
    pattern = re.compile(rf"(\n\s*{re.escape(patch_name)}\s*\{{)(.*?)(\n\s*\}})", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        body = match.group(2)
        body = re.sub(r"type\s+\w+\s*;", f"type            {patch_type};", body, count=1)
        body = re.sub(r"physicalType\s+\w+\s*;", f"physicalType    {patch_type};", body, count=1)
        return f"{match.group(1)}{body}{match.group(3)}"

    updated, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Patch {patch_name!r} not found in OpenFOAM boundary file")
    return updated


def patch_poly_boundary_types(case_dir: Path) -> None:
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    text = boundary_path.read_text(encoding="utf-8")
    patch_types = {
        "frontAndBack": "empty",
        "walls": "wall",
        "obstacle": "wall",
        "inlet": "patch",
        "outlet": "patch",
    }
    for patch_name, patch_type in patch_types.items():
        text = _replace_boundary_patch_type(text, patch_name, patch_type)
    boundary_path.write_text(text, encoding="utf-8")


def _write_geo(case_dir: Path, cfg: dict, outline: list[tuple[float, float]]) -> None:
    geom = cfd_geometry(cfg)
    mesh_cfg = cfg.get("cfd", {}).get("mesh", {})
    lc_far = float(mesh_cfg.get("lc_far_m", 0.012))
    lc_obstacle = float(mesh_cfg.get("lc_obstacle_m", 0.003))
    thickness = float(mesh_cfg.get("thickness_m", 0.01))

    lines = [
        'SetFactory("Built-in");',
        f"lcFar = {lc_far:.10g};",
        f"lcObs = {lc_obstacle:.10g};",
        f"thickness = {thickness:.10g};",
        "Point(1) = {0, 0, 0, lcFar};",
        f"Point(2) = {{{geom.tank_length_m:.10g}, 0, 0, lcFar}};",
        f"Point(3) = {{{geom.tank_length_m:.10g}, {geom.channel_height_m:.10g}, 0, lcFar}};",
        f"Point(4) = {{0, {geom.channel_height_m:.10g}, 0, lcFar}};",
        "Line(1) = {1, 2};",
        "Line(2) = {2, 3};",
        "Line(3) = {3, 4};",
        "Line(4) = {4, 1};",
    ]
    point_start = 100
    line_start = 100
    for idx, (x_coord, y_coord) in enumerate(outline):
        lines.append(f"Point({point_start + idx}) = {{{x_coord:.10g}, {y_coord:.10g}, 0, lcObs}};")
    obstacle_lines = []
    for idx in range(len(outline)):
        line_id = line_start + idx
        p0 = point_start + idx
        p1 = point_start + ((idx + 1) % len(outline))
        lines.append(f"Line({line_id}) = {{{p0}, {p1}}};")
        obstacle_lines.append(line_id)

    obstacle_loop = ", ".join(str(line_id) for line_id in obstacle_lines)
    obstacle_surfaces = ", ".join(f"out[{idx}]" for idx in range(6, 6 + len(outline)))
    lines.extend(
        [
            "Curve Loop(1) = {1, 2, 3, 4};",
            f"Curve Loop(2) = {{{obstacle_loop}}};",
            "Plane Surface(1) = {1, 2};",
            "out[] = Extrude {0, 0, thickness} { Surface{1}; Layers{1}; Recombine; };",
            'Physical Surface("walls") = {out[2], out[4]};',
            'Physical Surface("outlet") = {out[3]};',
            'Physical Surface("inlet") = {out[5]};',
            f'Physical Surface("obstacle") = {{{obstacle_surfaces}}};',
            'Physical Surface("frontAndBack") = {1, out[0]};',
            'Physical Volume("fluid") = {out[1]};',
            "Mesh.Algorithm = 6;",
            "Mesh.MshFileVersion = 2.2;",
        ]
    )
    (case_dir / "mesh.geo").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_prefix(openfoam_bashrc: str) -> str:
    bashrc = Path(openfoam_bashrc)
    if bashrc.exists():
        return f"source {bashrc} >/tmp/of_source.log 2>&1 || true; "
    return ""


def _run_shell(case_dir: Path, command: str, log_name: str, openfoam_bashrc: str = "") -> None:
    log_path = case_dir / "logs" / log_name
    shell_command = _source_prefix(openfoam_bashrc) + command
    with log_path.open("w", encoding="utf-8", errors="ignore") as handle:
        subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=case_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _run_mesh_commands(case_dir: Path, openfoam_bashrc: str) -> None:
    if shutil.which("gmsh") is None:
        raise RuntimeError("gmsh not found in PATH. Install gmsh or use --skip-mesh.")
    if shutil.which("gmshToFoam") is None:
        raise RuntimeError("gmshToFoam not found in PATH. Source OpenFOAM before building mesh.")
    _run_shell(case_dir, "gmsh -3 -format msh2 mesh.geo -o mesh.msh", "gmsh.log")
    _run_shell(case_dir, "gmshToFoam mesh.msh", "gmshToFoam.log", openfoam_bashrc)
    patch_poly_boundary_types(case_dir)
    _run_shell(case_dir, "checkMesh -constant", "checkMesh.log", openfoam_bashrc)


def build_case(
    case_dir: Path,
    cfg: dict,
    case: CfdCase,
    *,
    skip_mesh: bool,
    openfoam_bashrc: str = "/usr/share/openfoam/etc/bashrc",
) -> dict[str, object]:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for rel in ["0", "constant", "system", "logs"]:
        (case_dir / rel).mkdir(parents=True, exist_ok=True)

    geom = cfd_geometry(cfg)
    outline = projected_outline_m(
        case.stl_path, center_x_m=case.obstacle_x_m, center_y_m=case.obstacle_y_m
    )
    _write_geo(case_dir, cfg, [(float(x), float(y)) for x, y in outline])
    _write_control_dict(case_dir, cfg, case)
    _write_fv_files(case_dir)
    _write_fields(case_dir, case, nu_m2_s=geom.nu_m2_s)
    _write_change_dictionary(case_dir)
    _write_sample_dict(case_dir, cfg)

    payload = {
        "case": case.to_json(),
        "geometry": {
            "tank_length_m": geom.tank_length_m,
            "channel_height_m": geom.channel_height_m,
            "tank_width_m": geom.tank_width_m,
            "obstacle_area_m2": geom.obstacle_area_m2,
            "equivalent_diameter_m": geom.equivalent_diameter_m,
            "nu_m2_s": geom.nu_m2_s,
            "outline_points": int(outline.shape[0]),
        },
    }
    write_json(case_dir / "case_metadata.json", payload)

    status = "written"
    if not skip_mesh:
        _run_mesh_commands(case_dir, openfoam_bashrc=openfoam_bashrc)
        status = "meshed"
    return {
        "case_id": case.case_id,
        "shape": case.shape,
        "Re": case.re_value,
        "dy": case.dy_m,
        "split": case.split,
        "status": status,
        "case_dir": str(case_dir),
        "stl_path": case.stl_path,
        "inlet_u_m_s": case.inlet_u_m_s,
    }


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    output_root = Path(args.output_root).expanduser().resolve()
    case_root = output_root / "openfoam_cases"
    case_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in case_matrix(cfg, args.stl_dir):
        rows.append(
            build_case(
                case_root / case.case_id,
                cfg,
                case,
                skip_mesh=args.skip_mesh,
                openfoam_bashrc=args.openfoam_bashrc,
            )
        )

    index_path = output_root / "cfd_case_index.csv"
    pd.DataFrame(rows).sort_values("case_id").to_csv(index_path, index=False)
    print(f"Wrote {len(rows)} CFD cases to {case_root}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
