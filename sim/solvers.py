from __future__ import annotations

from pathlib import Path

from sim.openfoam_adapter import OpenFOAMSimulator
from sim.synthetic_solver import SyntheticSimulator


class BaseSimulator:
    def run_case(self, case_spec, out_dir: Path) -> Path:
        raise NotImplementedError


def build_simulator(solver_name: str, cfg: dict, logger):
    if solver_name == "synthetic":
        return SyntheticSimulator(cfg, logger)
    if solver_name == "openfoam":
        return OpenFOAMSimulator(cfg, logger)
    raise ValueError(f"Unknown solver: {solver_name}")
