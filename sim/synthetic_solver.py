from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sim.data_schema import PROBES_FILENAME, WAKE_FRAMES_FILENAME, write_metadata


class SyntheticSimulator:
    """Synthetic wake generator for both outlet probes and wake-field frame sequences."""

    def __init__(self, cfg: dict, logger):
        self.cfg = cfg
        self.logger = logger
        self.project_cfg = cfg["project"]
        self.sim_cfg = cfg["simulation"]
        self.time_cfg = self.sim_cfg["time"]
        self.synthetic_cfg = self.sim_cfg["synthetic"]
        self.challenge_cfg = self.synthetic_cfg.get("challenge", {})
        self.vision_cfg = cfg.get("vision", {})

        self.h = float(self.sim_cfg["H"])
        self.d = float(self.sim_cfg["d_ratio"]) * self.h
        self.u_mean = float(self.sim_cfg["U_mean"])
        self.n_probes = int(self.sim_cfg["probes_n"])
        self.l_in = float(self.sim_cfg["L_in"])
        self.l_out = float(self.sim_cfg["L_out"])
        self.x0 = float(self.sim_cfg["x0"])
        self.y0 = float(self.sim_cfg["y0"])
        self.l_total = self.l_in + self.l_out
        self.y_positions = (np.arange(self.n_probes, dtype=float) + 0.5) / self.n_probes * self.h

        self.modality = str(self.project_cfg.get("modality", "probe_signal"))
        self.sequence_frames = int(self.vision_cfg.get("sequence_frames", 8))
        self.field_size = int(self.vision_cfg.get("field_size", 128))
        self.channels = list(self.vision_cfg.get("channels", ["ux", "uy", "speed", "vorticity"]))

    def _shape_params(self, shape: str) -> dict:
        params = self.synthetic_cfg["shape_params"].get(shape)
        if params is None:
            raise ValueError(f"Unsupported shape for synthetic solver: {shape}")
        return params

    def _shape_profile(self, shape: str) -> dict[str, float]:
        profiles = {
            "circle": {
                "decay": 0.54,
                "spread": 0.070,
                "alt_offset": 0.14,
                "deficit": 0.22,
                "lift_bias": 0.00,
            },
            "square": {
                "decay": 0.62,
                "spread": 0.080,
                "alt_offset": 0.16,
                "deficit": 0.28,
                "lift_bias": 0.00,
            },
            "triangle": {
                "decay": 0.56,
                "spread": 0.074,
                "alt_offset": 0.13,
                "deficit": 0.24,
                "lift_bias": 0.00,
            },
            "airfoil": {
                "decay": 0.50,
                "spread": 0.060,
                "alt_offset": 0.10,
                "deficit": 0.18,
                "lift_bias": 0.05,
            },
            "diamond": {
                "decay": 0.60,
                "spread": 0.078,
                "alt_offset": 0.17,
                "deficit": 0.27,
                "lift_bias": 0.01,
            },
            "bar": {
                "decay": 0.70,
                "spread": 0.090,
                "alt_offset": 0.19,
                "deficit": 0.33,
                "lift_bias": 0.00,
            },
        }
        if shape not in profiles:
            raise ValueError(f"Unsupported shape: {shape}")
        return profiles[shape]

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
        return t_total, dt, n_transient

    @staticmethod
    def _base_profile(y_norm: np.ndarray) -> np.ndarray:
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
            mode = (
                0.58
                + 0.24 * np.cos(np.pi * (y_norm - 0.42))
                + 0.12 * np.sin(2.0 * np.pi * y_norm + 0.5)
            )
            return np.clip(mode, 0.15, None)
        if shape == "diamond":
            mode = (
                0.52
                + 0.28 * np.cos(2.0 * np.pi * (y_norm - 0.5))
                + 0.22 * np.sin(4.0 * np.pi * y_norm + 0.4)
            )
            return np.clip(mode, 0.12, None)
        if shape == "bar":
            center = np.exp(-((y_norm - 0.5) ** 2) / (2.0 * 0.10**2))
            shoulders = np.exp(-((y_norm - 0.32) ** 2) / (2.0 * 0.07**2)) + np.exp(
                -((y_norm - 0.68) ** 2) / (2.0 * 0.07**2)
            )
            mode = 0.45 + 0.55 * center + 0.25 * shoulders
            return np.clip(mode, 0.10, None)
        raise ValueError(f"Unsupported shape: {shape}")

    def _should_emit_wake_frames(self) -> bool:
        return self.modality == "wake_field"

    def _wake_canvas(self) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        eps_max = float(abs(self.sim_cfg["perturb"].get("eps_max", 0.0)))
        x_start = self.x0 + 0.5 * self.d
        x_end = self.l_total
        y_center = 0.5 * self.h
        h_canvas = self.h * (1.0 + eps_max)
        y_min = y_center - 0.5 * h_canvas
        y_max = y_center + 0.5 * h_canvas

        x = (np.arange(self.field_size, dtype=float) + 0.5) / self.field_size * (
            x_end - x_start
        ) + x_start
        y = (np.arange(self.field_size, dtype=float) + 0.5) / self.field_size * (
            y_max - y_min
        ) + y_min
        return x, y, x_start, x_end, y_min, y_max

    def _wake_velocity_field(
        self,
        case_spec,
        *,
        t: float,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        f0: float,
        amp_base: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self._shape_params(case_spec.shape)
        profile = self._shape_profile(case_spec.shape)

        x_start = self.x0 + 0.5 * self.d
        roi_len = max(1e-6, self.l_total - x_start)
        xn = np.clip((x_grid - x_start) / roi_len, 0.0, 1.0)

        x_transition = self.l_total - self.h
        frac = np.clip((x_grid - x_transition) / self.h, 0.0, 1.0)
        h_local = self.h * (1.0 + float(case_spec.eps) * frac)
        y_center = 0.5 * self.h
        y_bottom = y_center - 0.5 * h_local
        y_top = y_center + 0.5 * h_local
        inside = (y_grid >= y_bottom) & (y_grid <= y_top)

        y_norm = np.clip((y_grid - y_bottom) / (h_local + 1e-9), 1e-4, 1.0 - 1e-4)
        base_u = self.u_mean * self._base_profile(y_norm)

        lift_bias = profile["lift_bias"] * np.sqrt(float(case_spec.re) / 200.0)
        wake_center = self.y0 + float(case_spec.dy) + lift_bias * self.h * np.exp(-1.8 * xn)
        spread = profile["spread"] + 0.05 * xn + 0.02 * abs(float(case_spec.eps))
        cross = (y_grid - wake_center) / (self.h + 1e-9)
        core = np.exp(-(cross**2) / (2.0 * spread**2))
        offset = profile["alt_offset"] + 0.08 * xn
        sigma_pair = spread * 0.8
        upper = np.exp(-((cross - offset) ** 2) / (2.0 * sigma_pair**2))
        lower = np.exp(-((cross + offset) ** 2) / (2.0 * sigma_pair**2))
        pair_sum = upper + lower
        pair_diff = upper - lower

        decay = max(0.25, profile["decay"])
        wake_env = np.exp(-xn / decay)
        phase = (
            2.0 * np.pi * (0.65 * xn - f0 * t) + float(params.get("phase_gradient", 2.0)) * cross
        )
        harmonic = np.sin(phase)
        harmonic += float(params.get("h2", 0.0)) * np.sin(2.0 * phase + 0.3)
        harmonic += float(params.get("h3", 0.0)) * np.sin(3.0 * phase - 0.2)
        convective = np.cos(phase + 0.25) + 0.35 * np.cos(2.0 * phase - 0.15)

        deficit = profile["deficit"] * wake_env * np.clip(0.7 * core + 0.5 * pair_sum, 0.0, 1.6)
        u_field = base_u * (1.0 - deficit)
        u_field += 0.36 * amp_base * wake_env * (0.35 * core + pair_sum) * convective
        u_field += 0.18 * amp_base * wake_env * cross * np.sin(0.5 * phase + 0.4)

        v_field = 1.10 * amp_base * wake_env * pair_diff * harmonic
        v_field += 0.26 * amp_base * wake_env * core * np.sin(phase + 0.6)
        v_field += 0.08 * amp_base * wake_env * lift_bias

        u_field = np.where(inside, u_field, 0.0)
        v_field = np.where(inside, v_field, 0.0)

        mask = inside.astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.9, sigmaY=0.9)
        mask = np.clip(mask, 0.0, 1.0)
        return u_field.astype(np.float32), v_field.astype(np.float32), mask.astype(np.float32)

    @staticmethod
    def _particle_texture(
        rng: np.random.Generator, height: int, width: int, density: float, blur_sigma: float
    ) -> np.ndarray:
        texture = np.zeros((height, width), dtype=np.float32)
        dots = rng.random((height, width)) < density
        if np.any(dots):
            texture[dots] = rng.uniform(0.45, 1.0, size=int(np.sum(dots))).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        return np.clip(texture, 0.0, 1.0)

    def _render_wake_frames(
        self, case_spec, rng: np.random.Generator, t_sample: np.ndarray, f0: float, amp_base: float
    ) -> dict:
        if t_sample.size == 0:
            raise RuntimeError(
                "Synthetic solver produced no steady samples for wake frame rendering"
            )

        x_coords, y_coords, x_min, x_max, y_min, y_max = self._wake_canvas()
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        dt = float(self.time_cfg["dt"])
        frame_gap = float(np.clip(0.08 / max(f0, 0.08), 0.015, 0.05))
        stride = max(1, int(round(frame_gap / dt)))
        start_idx = max(0, t_sample.size - 1 - stride * (self.sequence_frames - 1))
        sample_indices = start_idx + stride * np.arange(self.sequence_frames, dtype=int)
        sample_indices = np.clip(sample_indices, 0, t_sample.size - 1)
        frame_times = t_sample[sample_indices]

        height = self.field_size
        width = self.field_size
        base_density = (
            0.055 + 0.02 * abs(float(case_spec.eps)) + 0.01 * (float(case_spec.re) / 300.0)
        )
        current = self._particle_texture(rng, height, width, density=base_density, blur_sigma=0.75)

        grid_x_pix, grid_y_pix = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        x_scale = width / max(1e-6, x_max - x_min)
        y_scale = height / max(1e-6, y_max - y_min)
        motion_gain = 0.85

        frames = []
        mask_ref: np.ndarray | None = None
        for frame_idx in range(self.sequence_frames):
            u_now, v_now, mask_now = self._wake_velocity_field(
                case_spec,
                t=float(frame_times[frame_idx]),
                x_grid=x_grid,
                y_grid=y_grid,
                f0=f0,
                amp_base=amp_base,
            )
            if mask_ref is None:
                mask_ref = mask_now
            frame = np.clip(
                current * mask_now + rng.normal(0.0, 0.012, size=current.shape), 0.0, 1.0
            )
            frame *= 0.92 + 0.12 * rng.random()
            frames.append(np.clip(frame, 0.0, 1.0).astype(np.float32))

            if frame_idx == self.sequence_frames - 1:
                continue

            dt_frame = float(frame_times[frame_idx + 1] - frame_times[frame_idx])
            x_disp = np.clip(u_now * dt_frame * x_scale * motion_gain, -3.0, 3.0).astype(np.float32)
            y_disp = np.clip(v_now * dt_frame * y_scale * motion_gain, -2.5, 2.5).astype(np.float32)
            map_x = grid_x_pix - x_disp
            map_y = grid_y_pix - y_disp
            warped = cv2.remap(
                current,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            injected = self._particle_texture(
                rng, height, width, density=base_density * 0.12, blur_sigma=0.65
            )
            warped = 0.95 * warped + 0.05 * injected
            warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=0.6, sigmaY=0.6)
            current = np.clip(
                warped * mask_now + rng.normal(0.0, 0.005, size=warped.shape), 0.0, 1.0
            ).astype(np.float32)

        if mask_ref is None:
            raise RuntimeError("Wake frame renderer failed to produce a reference mask")

        return {
            "frames": np.stack(frames, axis=0),
            "frame_times": frame_times.astype(np.float32),
            "sample_indices": sample_indices.astype(np.int32),
            "x_coords": x_coords.astype(np.float32),
            "y_coords": y_coords.astype(np.float32),
            "mask": mask_ref.astype(np.float32),
            "wake_roi": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "width": float(x_max - x_min),
                "height": float(y_max - y_min),
                "pixels_x": int(width),
                "pixels_y": int(height),
            },
        }

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

        common_amp = (
            float(self.challenge_cfg.get("common_mode_amp", 0.0)) if challenge_enabled else 0.0
        )
        common_freq_ratio = (
            float(self.challenge_cfg.get("common_mode_freq_ratio", 0.85))
            if challenge_enabled
            else 0.85
        )
        drift_amp = float(self.challenge_cfg.get("drift_amp", 0.0)) if challenge_enabled else 0.0
        drift_freq_ratio = (
            float(self.challenge_cfg.get("drift_freq_ratio", 0.08)) if challenge_enabled else 0.08
        )

        startup = 1.0 - np.exp(-t_total / max(1.0, float(self.time_cfg["transient_time"]) / 2.0))

        u_total = np.zeros((n_total, self.n_probes), dtype=float)
        for idx in range(self.n_probes):
            phi = 2.0 * np.pi * (phase_gradient * y_shifted[idx] + rng.uniform(0.0, 1.0))
            modulation = 1.0 + 0.03 * np.sin(2.0 * np.pi * 0.12 * f0 * t_total + 0.4 * phi)

            fundamental = np.sin(2.0 * np.pi * f0 * t_total * modulation + phi)
            harmonic2 = h2 * np.sin(2.0 * np.pi * 2.0 * f0 * t_total + 0.55 * phi)
            harmonic3 = h3 * np.sin(2.0 * np.pi * 3.0 * f0 * t_total + 0.25 * phi)
            broadband = 0.08 * np.sin(2.0 * np.pi * (f0 * 0.35) * t_total + 1.2 * phi)
            common_mode = common_amp * np.sin(
                2.0 * np.pi * (common_freq_ratio * f0) * t_total + 0.1 * phi
            )
            drift = drift_amp * np.sin(2.0 * np.pi * (drift_freq_ratio * f0) * t_total + 0.3 * phi)

            signal = (
                fundamental + harmonic2 + harmonic3 + broadband + common_mode + drift
            ) * startup
            noise = rng.normal(
                0.0, noise_std * (1.0 + 3.0 * abs(float(case_spec.eps))), size=n_total
            )

            u_total[:, idx] = (
                self.u_mean * base_u[idx] * lens_factor[idx] + amplitude[idx] * signal + noise
            )

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
                "x0": float(self.x0),
                "y0_nominal": float(self.y0),
                "y0_actual": float(self.y0) + float(case_spec.dy),
                "L_in": float(self.l_in),
                "L_out": float(self.l_out),
                "L_total": float(self.l_total),
            },
            "probes": {
                "count": int(self.n_probes),
                "x": float(self.l_total),
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

        if self._should_emit_wake_frames():
            wake_payload = self._render_wake_frames(
                case_spec, rng, t_sample=t_sample, f0=f0, amp_base=amp_base
            )
            wake_frames_path = out_dir / WAKE_FRAMES_FILENAME
            np.savez_compressed(
                wake_frames_path,
                frames=wake_payload["frames"],
                frame_times=wake_payload["frame_times"],
                sample_indices=wake_payload["sample_indices"],
                x_coords=wake_payload["x_coords"],
                y_coords=wake_payload["y_coords"],
                mask=wake_payload["mask"],
            )
            metadata["steady_frame_range"] = {
                "sample_index_start": int(wake_payload["sample_indices"][0]),
                "sample_index_end": int(wake_payload["sample_indices"][-1]),
                "t_start": float(wake_payload["frame_times"][0]),
                "t_end": float(wake_payload["frame_times"][-1]),
                "n_frames": int(self.sequence_frames),
            }
            metadata["wake_roi"] = wake_payload["wake_roi"]
            metadata["files"]["wake_frames_npz"] = WAKE_FRAMES_FILENAME

        write_metadata(out_dir, metadata)
        return output_csv
