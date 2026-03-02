from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim.data_schema import PROBES_FILENAME, write_metadata


class SyntheticSimulator:
    """Synthetic wake signal generator that mimics outlet probe velocity signals."""

    def __init__(self, cfg: dict, logger):
        self.cfg = cfg
        self.logger = logger
        self.sim_cfg = cfg["simulation"]
        self.time_cfg = self.sim_cfg["time"]
        self.synthetic_cfg = self.sim_cfg["synthetic"]
        self.challenge_cfg = self.synthetic_cfg.get("challenge", {})

        self.h = float(self.sim_cfg["H"])
        self.d = float(self.sim_cfg["d_ratio"]) * self.h
        self.u_mean = float(self.sim_cfg["U_mean"])
        self.n_probes = int(self.sim_cfg["probes_n"])
        self.l_in = float(self.sim_cfg["L_in"])
        self.l_out = float(self.sim_cfg["L_out"])
        self.y_positions = (np.arange(self.n_probes, dtype=float) + 0.5) / self.n_probes * self.h

    def _shape_params(self, shape: str) -> dict:
        params = self.synthetic_cfg["shape_params"].get(shape)
        if params is None:
            raise ValueError(f"Unsupported shape for synthetic solver: {shape}")
        return params

    def _main_frequency(self, case_spec) -> float:
        params = self._shape_params(case_spec.shape)
        st = float(params["st"])
        base = st * self.u_mean / self.d
        re_factor = 1.0 + 0.08 * ((float(case_spec.re) - 200.0) / 200.0)
        geom_factor = 1.0 + 0.25 * float(case_spec.eps) + 0.10 * (float(case_spec.dy) / self.h)
        return max(0.05, base * re_factor * geom_factor)

    def _time_axis(self, f0: float) -> tuple[np.ndarray, float, int]:
        dt = float(self.time_cfg["dt"])
        min_cycles = float(self.time_cfg["min_cycles"])
        min_samples = int(self.time_cfg["min_samples"])
        transient_time = float(self.time_cfg["transient_time"])
        transient_cycles = float(self.time_cfg.get("transient_cycles", 6.0))

        sample_duration = max(min_cycles / f0, min_samples * dt)
        transient_duration = max(transient_time, transient_cycles / f0)

        n_transient = int(np.ceil(transient_duration / dt))
        n_sample = int(np.ceil(sample_duration / dt))
        n_total = n_transient + n_sample

        t_total = np.arange(n_total, dtype=float) * dt
        t_sample = np.arange(n_sample, dtype=float) * dt
        return t_total, dt, n_transient

    @staticmethod
    def _base_profile(y_norm: np.ndarray) -> np.ndarray:
        # Parabolic profile whose cross-section average equals U_mean when scaled.
        return 6.0 * y_norm * (1.0 - y_norm)

    @staticmethod
    def _shape_spatial_mode(shape: str, y_norm: np.ndarray) -> np.ndarray:
        if shape == "circle":
            return 0.7 + 0.3 * np.cos(np.pi * (y_norm - 0.5))
        if shape == "square":
            return 0.6 + 0.4 * np.sin(np.pi * y_norm) ** 2
        if shape == "triangle":
            return 0.55 + 0.45 * np.cos(2.0 * np.pi * y_norm + 0.6)
        if shape == "airfoil":
            # Skewed wake mode to mimic lift-induced asymmetry.
            mode = (
                0.58
                + 0.24 * np.cos(np.pi * (y_norm - 0.42))
                + 0.12 * np.sin(2.0 * np.pi * y_norm + 0.5)
            )
            return np.clip(mode, 0.15, None)
        raise ValueError(f"Unsupported shape: {shape}")

    def run_case(self, case_spec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(case_spec.seed))

        params = self._shape_params(case_spec.shape)
        f0 = self._main_frequency(case_spec)

        challenge_enabled = bool(self.challenge_cfg.get("enabled", False))
        if challenge_enabled:
            freq_jitter_std = float(self.challenge_cfg.get("freq_jitter_std", 0.0))
            f0 *= float(np.clip(1.0 + rng.normal(0.0, freq_jitter_std), 0.7, 1.3))

        t_total, dt, n_transient = self._time_axis(f0)
        t_sample = t_total[n_transient:] - t_total[n_transient]
        n_total = t_total.size

        y_shifted = np.clip((self.y_positions - float(case_spec.dy)) / self.h, 1e-4, 1.0 - 1e-4)
        base_u = self._base_profile(y_shifted)
        lens_factor = 1.0 - 0.9 * float(case_spec.eps) * (y_shifted - 0.5)
        lens_factor = np.clip(lens_factor, 0.85, 1.15)

        amp_base = float(params["amp"]) * np.sqrt(float(case_spec.re) / 200.0)
        if challenge_enabled:
            amp_jitter_std = float(self.challenge_cfg.get("amp_jitter_std", 0.0))
            amp_base *= float(np.clip(1.0 + rng.normal(0.0, amp_jitter_std), 0.6, 1.6))
        spatial_amp = self._shape_spatial_mode(case_spec.shape, y_shifted)
        amplitude = amp_base * spatial_amp * (1.0 + 1.5 * abs(float(case_spec.eps)))

        h2 = float(params.get("h2", 0.25))
        h3 = float(params.get("h3", 0.10))
        phase_gradient = float(params.get("phase_gradient", 2.0))
        noise_std = float(self.synthetic_cfg["noise_std"])
        if challenge_enabled:
            noise_std *= float(self.challenge_cfg.get("noise_multiplier", 1.0))

        common_amp = float(self.challenge_cfg.get("common_mode_amp", 0.0)) if challenge_enabled else 0.0
        common_freq_ratio = float(self.challenge_cfg.get("common_mode_freq_ratio", 0.85)) if challenge_enabled else 0.85
        drift_amp = float(self.challenge_cfg.get("drift_amp", 0.0)) if challenge_enabled else 0.0
        drift_freq_ratio = float(self.challenge_cfg.get("drift_freq_ratio", 0.08)) if challenge_enabled else 0.08

        startup = 1.0 - np.exp(-t_total / max(1.0, float(self.time_cfg["transient_time"]) / 2.0))

        u_total = np.zeros((n_total, self.n_probes), dtype=float)
        for idx in range(self.n_probes):
            phi = 2.0 * np.pi * (phase_gradient * y_shifted[idx] + rng.uniform(0.0, 1.0))
            modulation = 1.0 + 0.03 * np.sin(2.0 * np.pi * 0.12 * f0 * t_total + 0.4 * phi)

            fundamental = np.sin(2.0 * np.pi * f0 * t_total * modulation + phi)
            harmonic2 = h2 * np.sin(2.0 * np.pi * 2.0 * f0 * t_total + 0.55 * phi)
            harmonic3 = h3 * np.sin(2.0 * np.pi * 3.0 * f0 * t_total + 0.25 * phi)
            broadband = 0.08 * np.sin(2.0 * np.pi * (f0 * 0.35) * t_total + 1.2 * phi)
            common_mode = common_amp * np.sin(2.0 * np.pi * (common_freq_ratio * f0) * t_total + 0.1 * phi)
            drift = drift_amp * np.sin(2.0 * np.pi * (drift_freq_ratio * f0) * t_total + 0.3 * phi)

            signal = (fundamental + harmonic2 + harmonic3 + broadband + common_mode + drift) * startup
            noise = rng.normal(0.0, noise_std * (1.0 + 3.0 * abs(float(case_spec.eps))), size=n_total)

            u_total[:, idx] = self.u_mean * base_u[idx] * lens_factor[idx] + amplitude[idx] * signal + noise

        if challenge_enabled:
            probe_mix = float(self.challenge_cfg.get("probe_mix", 0.0))
            if probe_mix > 0.0:
                u_total = (
                    (1.0 - probe_mix) * u_total
                    + 0.5 * probe_mix * np.roll(u_total, shift=1, axis=1)
                    + 0.5 * probe_mix * np.roll(u_total, shift=-1, axis=1)
                )

            dropout_prob = float(self.challenge_cfg.get("dropout_prob", 0.0))
            dropout_std = float(self.challenge_cfg.get("dropout_std", 0.0))
            if dropout_prob > 0.0 and dropout_std > 0.0:
                mask = rng.random(u_total.shape) < dropout_prob
                u_total[mask] += rng.normal(0.0, dropout_std, size=int(np.sum(mask)))

        u_sample = u_total[n_transient:, :]
        columns = [f"u_{i:03d}" for i in range(self.n_probes)]
        df = pd.DataFrame(u_sample, columns=columns)
        df.insert(0, "time", t_sample)

        output_csv = out_dir / PROBES_FILENAME
        df.to_csv(output_csv, index=False)

        metadata = {
            "case_id": case_spec.case_id,
            "backend": "synthetic",
            "shape": case_spec.shape,
            "Re": int(case_spec.re),
            "dy": float(case_spec.dy),
            "eps": float(case_spec.eps),
            "seed": int(case_spec.seed),
            "geometry": {
                "H": float(self.h),
                "d": float(self.d),
                "x0": float(self.sim_cfg["x0"]),
                "y0_nominal": float(self.sim_cfg["y0"]),
                "y0_actual": float(self.sim_cfg["y0"]) + float(case_spec.dy),
                "L_in": float(self.l_in),
                "L_out": float(self.l_out),
                "L_total": float(self.l_in + self.l_out),
            },
            "probes": {
                "count": int(self.n_probes),
                "x": float(self.l_in + self.l_out),
                "y": [float(v) for v in self.y_positions],
                "components": ["u"],
            },
            "sampling": {
                "dt": float(dt),
                "n_samples": int(df.shape[0]),
                "t_start": float(t_sample[0]) if t_sample.size else 0.0,
                "t_end": float(t_sample[-1]) if t_sample.size else 0.0,
                "transient_steps": int(n_transient),
                "f0_est": float(f0),
            },
            "synthetic_profile": {
                "challenge_enabled": challenge_enabled,
                "noise_std": float(noise_std),
            },
            "files": {
                "probes_csv": PROBES_FILENAME,
            },
        }
        write_metadata(out_dir, metadata)

        return output_csv
