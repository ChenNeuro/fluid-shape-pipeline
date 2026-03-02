from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    shape: str
    re: int
    sample_idx: int
    seed: int
    dy: float
    eps: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_case_specs(cfg: dict) -> list[CaseSpec]:
    sim_cfg = cfg["simulation"]
    perturb_cfg = sim_cfg["perturb"]
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)

    h = float(sim_cfg["H"])
    shapes = list(sim_cfg["shapes"])
    re_values = [int(v) for v in sim_cfg["re_values"]]
    samples_per_combo = int(sim_cfg["perturbations_per_combo"])

    cases: list[CaseSpec] = []
    for shape in shapes:
        for re in re_values:
            for sample_idx in range(samples_per_combo):
                if perturb_cfg.get("enable_dy", True):
                    dy_min = float(perturb_cfg.get("dy_min", -0.02))
                    dy_max = float(perturb_cfg.get("dy_max", 0.02))
                    dy = float(rng.uniform(dy_min, dy_max)) * h
                else:
                    dy = 0.0

                if perturb_cfg.get("enable_eps", True):
                    eps_max = float(perturb_cfg.get("eps_max", 0.02))
                    eps_mag = float(rng.uniform(0.0, eps_max))
                    eps_sign = -1.0 if rng.random() < 0.5 else 1.0
                    eps = eps_sign * eps_mag
                else:
                    eps = 0.0

                case_seed = int(rng.integers(0, 2**31 - 1))
                case_id = f"{shape}_Re{re}_p{sample_idx:02d}"
                cases.append(
                    CaseSpec(
                        case_id=case_id,
                        shape=shape,
                        re=re,
                        sample_idx=sample_idx,
                        seed=case_seed,
                        dy=dy,
                        eps=eps,
                    )
                )

    return cases
