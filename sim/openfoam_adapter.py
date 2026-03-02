from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from sim.config import resolve_from_root
from sim.data_schema import PROBES_FILENAME, write_metadata


class OpenFOAMSimulator:
    """Adapter to run OpenFOAM and export probe u(t) CSV in the same schema as synthetic solver."""

    def __init__(self, cfg: dict, logger):
        self.cfg = cfg
        self.logger = logger
        self.sim_cfg = cfg["simulation"]
        self.of_cfg = cfg["openfoam"]

        self.template_dir = resolve_from_root(self.of_cfg["template_dir"])
        self.case_root = resolve_from_root(self.of_cfg["case_root"])
        self.n_probes = int(self.sim_cfg["probes_n"])
        self.h = float(self.sim_cfg["H"])
        self.l_in = float(self.sim_cfg["L_in"])
        self.l_out = float(self.sim_cfg["L_out"])
        self.x_probe = self.l_in + self.l_out

    def _required_commands(self) -> list[str]:
        cmds = list(self.of_cfg.get("mesh_cmds", [])) + [self.of_cfg.get("solver_cmd", "pimpleFoam")]
        return [cmd for cmd in cmds if cmd]

    def _check_openfoam_commands(self) -> None:
        missing = [cmd for cmd in self._required_commands() if shutil.which(cmd) is None]
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                f"OpenFOAM commands not found in PATH: {missing_text}. "
                "Use solver=synthetic or install/source OpenFOAM environment first."
            )

    def _prepare_workdir(self, case_id: str) -> Path:
        work_dir = self.case_root / case_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(self.template_dir, work_dir)
        return work_dir

    def _probe_block(self) -> str:
        n = int(self.sim_cfg["probes_n"])
        entries = []
        for i in range(n):
            y = (i + 0.5) / n * self.h
            entries.append(f"        ({self.x_probe:.6f} {y:.6f} 0)")
        return "\n".join(entries)

    def _probe_positions_y(self) -> list[float]:
        n = int(self.sim_cfg["probes_n"])
        return [float((i + 0.5) / n * self.h) for i in range(n)]

    def _render_templates(self, work_dir: Path, case_spec) -> None:
        u_mean = float(self.sim_cfg["U_mean"])
        h = float(self.sim_cfg["H"])
        d = float(self.sim_cfg["d_ratio"]) * h
        l_in = float(self.sim_cfg["L_in"])
        l_out = float(self.sim_cfg["L_out"])
        l_total = l_in + l_out
        x0 = float(self.sim_cfg["x0"])
        y0 = float(self.sim_cfg["y0"]) + float(case_spec.dy)
        nu = u_mean * d / float(case_spec.re)

        replacements = {
            "__END_TIME__": str(float(self.sim_cfg["time"]["transient_time"]) + float(self.sim_cfg["time"]["min_samples"]) * float(self.sim_cfg["time"]["dt"])),
            "__DELTA_T__": str(float(self.sim_cfg["time"]["dt"])),
            "__WRITE_INTERVAL__": str(float(self.of_cfg.get("write_interval", 0.1))),
            "__PROBE_POINTS__": self._probe_block(),
            "__U_MEAN__": str(u_mean),
            "__NU__": str(nu),
            "__L__": str(l_total),
            "__L_IN__": str(l_in),
            "__L_OUT__": str(l_out),
            "__H__": str(h),
            "__X0__": str(x0),
            "__Y0__": str(y0),
            "__D__": str(d),
        }

        for rel_path in [
            "system/controlDict",
            "system/blockMeshDict",
            "0/U",
            "constant/transportProperties",
        ]:
            file_path = work_dir / rel_path
            text = file_path.read_text(encoding="utf-8")
            for key, value in replacements.items():
                text = text.replace(key, value)
            file_path.write_text(text, encoding="utf-8")

        metadata = {
            "case_id": case_spec.case_id,
            "shape": case_spec.shape,
            "Re": int(case_spec.re),
            "dy": float(case_spec.dy),
            "eps": float(case_spec.eps),
            "seed": int(case_spec.seed),
            "note": "Geometry perturbation hooks are prepared in this adapter. Extend mesh generation scripts for exact obstacle/lens geometry.",
        }
        with (work_dir / "case_params.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _run_geometry_hook(self, work_dir: Path) -> None:
        hook_script = work_dir / "scripts" / "build_geometry.py"
        if not hook_script.exists():
            return
        python_bin = shutil.which("python") or "python"
        subprocess.run([python_bin, str(hook_script), "--case-dir", str(work_dir)], check=True)

    @staticmethod
    def _parse_vector_probe_file(probe_file: Path, n_probes: int) -> pd.DataFrame:
        rows: list[list[float]] = []
        vector_pattern = re.compile(r"\(([^\)]+)\)")

        with probe_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                pieces = stripped.split(maxsplit=1)
                if len(pieces) < 2:
                    continue

                try:
                    time_value = float(pieces[0])
                except ValueError:
                    continue

                vectors = vector_pattern.findall(pieces[1])
                if not vectors:
                    continue

                u_values = []
                for token in vectors[:n_probes]:
                    comps = token.split()
                    if not comps:
                        continue
                    u_values.append(float(comps[0]))

                if len(u_values) != n_probes:
                    continue

                rows.append([time_value, *u_values])

        if not rows:
            raise RuntimeError(f"No probe samples parsed from {probe_file}")

        columns = ["time", *[f"u_{i:03d}" for i in range(n_probes)]]
        return pd.DataFrame(rows, columns=columns)

    def _run_commands(self, work_dir: Path) -> None:
        for cmd in self._required_commands():
            self.logger.info("Running OpenFOAM command '%s' in %s", cmd, work_dir)
            subprocess.run(cmd, cwd=work_dir, shell=True, check=True)

    def _find_probe_output(self, work_dir: Path) -> Path:
        candidates = sorted((work_dir / "postProcessing" / "probes").glob("*/U"))
        if not candidates:
            raise RuntimeError(f"No probe output found in {work_dir / 'postProcessing/probes'}")
        return candidates[-1]

    def run_case(self, case_spec, out_dir: Path) -> Path:
        self._check_openfoam_commands()

        if not self.template_dir.exists():
            raise RuntimeError(f"OpenFOAM template not found: {self.template_dir}")

        work_dir = self._prepare_workdir(case_spec.case_id)
        self._render_templates(work_dir, case_spec)
        self._run_geometry_hook(work_dir)
        self._run_commands(work_dir)

        probe_file = self._find_probe_output(work_dir)
        df = self._parse_vector_probe_file(probe_file, self.n_probes)

        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / PROBES_FILENAME
        df.to_csv(output_csv, index=False)

        metadata = {
            "case_id": case_spec.case_id,
            "backend": "openfoam",
            "shape": case_spec.shape,
            "Re": int(case_spec.re),
            "dy": float(case_spec.dy),
            "eps": float(case_spec.eps),
            "seed": int(case_spec.seed),
            "geometry": {
                "H": float(self.h),
                "d": float(float(self.sim_cfg["d_ratio"]) * self.h),
                "x0": float(self.sim_cfg["x0"]),
                "y0_nominal": float(self.sim_cfg["y0"]),
                "y0_actual": float(self.sim_cfg["y0"]) + float(case_spec.dy),
                "L_in": float(self.l_in),
                "L_out": float(self.l_out),
                "L_total": float(self.l_in + self.l_out),
            },
            "probes": {
                "count": int(self.n_probes),
                "x": float(self.x_probe),
                "y": self._probe_positions_y(),
                "components": ["u"],
            },
            "sampling": {
                "dt": float(df["time"].diff().dropna().median()) if df.shape[0] > 1 else None,
                "n_samples": int(df.shape[0]),
                "t_start": float(df["time"].iloc[0]) if df.shape[0] > 0 else None,
                "t_end": float(df["time"].iloc[-1]) if df.shape[0] > 0 else None,
            },
            "files": {
                "probes_csv": PROBES_FILENAME,
                "source_probe_file": str(probe_file),
            },
        }
        write_metadata(out_dir, metadata)
        return output_csv
