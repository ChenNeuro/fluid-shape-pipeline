from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from scripts.cfd.common import latest_time_dir, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prepared OpenFOAM CFD cases")
    parser.add_argument("--case-root", required=True, help="Directory containing OpenFOAM cases")
    parser.add_argument("--workers", type=int, default=1, help="Number of cases to run in parallel")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for smoke runs")
    parser.add_argument("--only", default=None, help="Optional case_id regex")
    parser.add_argument(
        "--openfoam-bashrc",
        default="/usr/share/openfoam/etc/bashrc",
        help="OpenFOAM bashrc to source before running commands",
    )
    return parser.parse_args()


def _source_prefix(openfoam_bashrc: str) -> str:
    bashrc = Path(openfoam_bashrc)
    if bashrc.exists():
        return f"source {bashrc} >/tmp/of_source.log 2>&1 || true; "
    return ""


def _run_shell(case_dir: Path, command: str, log_path: Path, openfoam_bashrc: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_command = _source_prefix(openfoam_bashrc) + command
    with log_path.open("w", encoding="utf-8", errors="ignore") as handle:
        completed = subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=case_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _latest_log_contains(case_dir: Path, patterns: list[str]) -> bool:
    text_parts = []
    for log_path in sorted((case_dir / "logs").glob("*.log")):
        text_parts.append(log_path.read_text(encoding="utf-8", errors="ignore")[-20000:])
    text = "\n".join(text_parts).lower()
    return any(pattern.lower() in text for pattern in patterns)


def _parse_mesh_cells(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"cells:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def _case_dirs(case_root: Path, only: str | None, max_cases: int | None) -> list[Path]:
    candidates = sorted(
        path for path in case_root.iterdir() if (path / "case_metadata.json").exists()
    )
    if only:
        pattern = re.compile(only)
        candidates = [path for path in candidates if pattern.search(path.name)]
    if max_cases is not None:
        candidates = candidates[:max_cases]
    return candidates


def _ensure_mesh(case_dir: Path, openfoam_bashrc: str) -> tuple[bool, str]:
    if (case_dir / "constant" / "polyMesh").exists():
        return True, "existing"
    if shutil.which("gmsh") is None:
        return False, "gmsh_missing"
    if not Path(openfoam_bashrc).exists() and shutil.which("gmshToFoam") is None:
        return False, "openfoam_missing"

    commands = [
        ("gmsh", "gmsh -3 -format msh2 mesh.geo -o mesh.msh"),
        ("gmshToFoam", "gmshToFoam mesh.msh"),
        ("changeDictionary", "changeDictionary"),
    ]
    for name, command in commands:
        code = _run_shell(case_dir, command, case_dir / "logs" / f"{name}.log", openfoam_bashrc)
        if code != 0:
            return False, f"{name}_failed"
    return True, "created"


def run_case(case_dir: Path, openfoam_bashrc: str) -> dict[str, object]:
    meta = read_json(case_dir / "case_metadata.json")
    case = meta["case"]
    row: dict[str, object] = {
        "case_id": case["case_id"],
        "shape": case["shape"],
        "Re": case["re_value"],
        "dy": case["dy_m"],
        "split": case["split"],
        "case_dir": str(case_dir),
    }

    mesh_ok, mesh_status = _ensure_mesh(case_dir, openfoam_bashrc)
    row["mesh_status"] = mesh_status
    if not mesh_ok:
        row["status"] = "mesh_failed"
        return row

    check_code = _run_shell(
        case_dir, "checkMesh -constant", case_dir / "logs" / "checkMesh.log", openfoam_bashrc
    )
    row["checkMesh_returncode"] = check_code
    row["mesh_cells"] = _parse_mesh_cells(case_dir / "logs" / "checkMesh.log")
    if check_code != 0:
        row["status"] = "checkMesh_failed"
        return row

    solve_code = _run_shell(
        case_dir, "pimpleFoam", case_dir / "logs" / "pimpleFoam.log", openfoam_bashrc
    )
    row["pimpleFoam_returncode"] = solve_code
    if solve_code != 0:
        row["status"] = "solver_failed"
        return row

    sample_code = _run_shell(
        case_dir,
        "postProcess -func writeCellCentres -latestTime",
        case_dir / "logs" / "sample.log",
        openfoam_bashrc,
    )
    row["sample_returncode"] = sample_code
    final_time_dir = latest_time_dir(case_dir)
    row["final_time"] = final_time_dir.name if final_time_dir else None
    row["bad_log_signal"] = _latest_log_contains(
        case_dir,
        ["foam fatal", "negative volume", "divergence detected"],
    )
    row["status"] = (
        "success" if sample_code == 0 and not row["bad_log_signal"] else "postprocess_failed"
    )
    return row


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root).expanduser().resolve()
    cases = _case_dirs(case_root, only=args.only, max_cases=args.max_cases)
    if not cases:
        raise FileNotFoundError(f"No cases found in {case_root}")

    workers = max(1, int(args.workers))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, case_dir, args.openfoam_bashrc) for case_dir in cases]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['case_id']}: {row['status']}")

    health_path = case_root.parent / "cfd_case_health.csv"
    pd.DataFrame(rows).sort_values("case_id").to_csv(health_path, index=False)
    print(f"Health: {health_path}")


if __name__ == "__main__":
    main()
