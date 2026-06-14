from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cfd.build_openfoam_cases import build_case
from scripts.cfd.build_wake_fields_from_openfoam import build_case_wake_field
from scripts.cfd.common import CfdCase
from vision.wake_field_builder import build_crop_boxes


def _cfg() -> dict:
    return {
        "project": {"seed": 1, "solver": "openfoam", "modality": "cfd_wake_field"},
        "simulation": {
            "H": 0.45,
            "d_ratio": 0.1406,
            "x0": 0.25,
            "y0": 0.225,
            "L_in": 0.25,
            "L_out": 0.60,
            "re_values": [800],
            "shapes": ["circle"],
        },
        "cfd": {
            "tank": {"length_m": 0.85, "depth_m": 0.45, "width_m": 0.40},
            "obstacle": {
                "area_m2": 0.005,
                "equivalent_diameter_m": 0.079788456,
                "center_x_m": 0.25,
                "center_y_m": 0.225,
            },
            "fluid": {"nu_m2_s": 1e-6},
            "mesh": {"thickness_m": 0.01, "lc_far_m": 0.05, "lc_obstacle_m": 0.01},
            "time": {"dt_s": 0.01, "end_time_s": 0.02, "write_interval_s": 0.02},
        },
        "vision": {
            "field_size": 16,
            "channels": ["ux", "uy", "speed", "vorticity"],
            "scales": ["distD1.0_full", "distD2.0_full", "distD4.0_full"],
        },
    }


def _write_ascii_stl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "solid box",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex 80 0 0",
                "vertex 80 80 0",
                "endloop",
                "endfacet",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex 80 80 0",
                "vertex 0 80 0",
                "endloop",
                "endfacet",
                "endsolid box",
            ]
        ),
        encoding="utf-8",
    )


class CfdPipelineTests(unittest.TestCase):
    def test_dist_d_crop_starts_from_obstacle_diameter(self) -> None:
        boxes = build_crop_boxes(
            np.zeros((16, 16), dtype=np.float32),
            ["distD2.0_full"],
            physical_h=0.45,
            physical_obstacle_d=0.08,
            canvas_x_start=0.29,
            canvas_x_end=0.85,
            canvas_y_min=0.0,
            canvas_y_max=0.45,
        )
        expected_x0 = (0.16) / (0.85 - 0.29)
        self.assertAlmostEqual(boxes["distD2.0_full"][0], expected_x0, places=6)
        self.assertEqual(boxes["distD2.0_full"][1:], [0.0, 1.0, 1.0])

    def test_build_case_writes_openfoam_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stl = root / "circle_5000mm2.STL"
            _write_ascii_stl(stl)
            case = CfdCase(
                case_id="circle_Re800_dy0",
                shape="circle",
                re_value=800,
                dy_m=0.0,
                split="test",
                inlet_u_m_s=0.01,
                obstacle_x_m=0.25,
                obstacle_y_m=0.225,
                stl_path=str(stl),
            )
            row = build_case(root / "case", _cfg(), case, skip_mesh=True)
            self.assertEqual(row["status"], "written")
            self.assertTrue((root / "case" / "mesh.geo").exists())
            self.assertTrue((root / "case" / "0" / "U").exists())
            self.assertIn(
                'Physical Surface("obstacle")',
                (root / "case" / "mesh.geo").read_text(encoding="utf-8"),
            )

    def test_build_wake_field_from_sample_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stl = root / "circle_5000mm2.STL"
            _write_ascii_stl(stl)
            case = CfdCase(
                case_id="circle_Re800_dy0",
                shape="circle",
                re_value=800,
                dy_m=0.0,
                split="test",
                inlet_u_m_s=0.01,
                obstacle_x_m=0.25,
                obstacle_y_m=0.225,
                stl_path=str(stl),
            )
            case_dir = root / "case"
            build_case(case_dir, _cfg(), case, skip_mesh=True)
            sample_dir = case_dir / "postProcessing" / "sample" / "1"
            sample_dir.mkdir(parents=True)
            rows = []
            for y_value in np.linspace(0.0, 0.45, 8):
                for x_value in np.linspace(0.30, 0.85, 8):
                    rows.append(
                        f"{x_value} {y_value} 0 {0.01 + x_value * 0.001} {y_value * 0.001} 0"
                    )
            (sample_dir / "wakePlane_U.raw").write_text("\n".join(rows), encoding="utf-8")

            row = build_case_wake_field(
                case_dir=case_dir,
                cfg=_cfg(),
                run_raw_dir=root / "run" / "data" / "raw",
                openfoam_bashrc="/missing/openfoam/bashrc",
            )
            self.assertEqual(row["domain"], "cfd")
            payload = np.load(root / "run" / "data" / "raw" / "circle_Re800_dy0" / "wake_field.npz")
            self.assertEqual(payload["crops"].shape, (3, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
