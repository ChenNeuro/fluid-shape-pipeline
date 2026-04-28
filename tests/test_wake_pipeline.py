from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vision.wake_field_builder import build_distance_crop_box


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "wake_field_smoke.yaml"


def _run_smoke_pipeline(temp_root: Path) -> None:
    env = dict(os.environ)
    env["FLUID_SHAPE_PIPELINE_ROOT"] = str(temp_root)
    commands = [
        [sys.executable, "-m", "sim.generate_dataset", "--config", str(SMOKE_CONFIG)],
        [sys.executable, "-m", "extract.build_wake_fields", "--config", str(SMOKE_CONFIG)],
        [sys.executable, "-m", "ml.train_wake", "--config", str(SMOKE_CONFIG)],
        [sys.executable, "-m", "ml.reconstruct_wake", "--config", str(SMOKE_CONFIG)],
    ]

    for cmd in commands:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


class WakePipelineSmokeTest(unittest.TestCase):
    def test_end_to_end_smoke_pipeline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wake-pipeline-smoke-") as tmpdir:
            temp_root = Path(tmpdir)
            _run_smoke_pipeline(temp_root)

            manifest_path = temp_root / "data" / "raw" / "manifest.csv"
            self.assertTrue(manifest_path.exists(), manifest_path)

            wake_index_path = temp_root / "data" / "wake_fields" / "index.csv"
            self.assertTrue(wake_index_path.exists(), wake_index_path)

            wake_index = pd.read_csv(wake_index_path).sort_values("case_id").reset_index(drop=True)
            first_case_id = str(wake_index.loc[0, "case_id"])
            case_dir = temp_root / "data" / "raw" / first_case_id

            frames_path = case_dir / "wake_frames.npz"
            field_path = case_dir / "wake_field.npz"
            metadata_path = case_dir / "metadata.json"
            self.assertTrue(frames_path.exists(), frames_path)
            self.assertTrue(field_path.exists(), field_path)
            self.assertTrue(metadata_path.exists(), metadata_path)

            smoke_cfg = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
            expected_scales = [str(item) for item in smoke_cfg["vision"]["scales"]]
            expected_channels = [str(item) for item in smoke_cfg["vision"]["channels"]]

            field_payload = np.load(field_path)
            self.assertEqual(int(field_payload["field_raw"].shape[0]), len(expected_channels))
            self.assertEqual(int(field_payload["crops"].shape[0]), len(expected_scales))
            self.assertEqual(int(field_payload["crops"].shape[1]), len(expected_channels))
            self.assertEqual([str(item) for item in field_payload["scales"].tolist()], expected_scales)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("field_channels", metadata)
            self.assertIn("crop_boxes", metadata)
            self.assertEqual(sorted(metadata["crop_boxes"]), sorted(expected_scales))
            for scale_name in expected_scales:
                box = metadata["crop_boxes"][scale_name]
                self.assertGreaterEqual(box[0], 0.0)
                self.assertGreaterEqual(box[1], 0.0)
                self.assertLessEqual(box[2], 1.0)
                self.assertLessEqual(box[3], 1.0)

            model_names = {path.name for path in (temp_root / "models").iterdir()}
            self.assertEqual(model_names, {"wake_field_main.pt", "wake_field_single.pt"})

            selection_payload = json.loads((temp_root / "reports" / "wake_field_selection.json").read_text(encoding="utf-8"))
            self.assertEqual(selection_payload["main_variant"], "dist_multi_4ch")
            self.assertIsNone(selection_payload["speed_variant"])

            summary_text = (temp_root / "reports" / "wake_field_summary.md").read_text(encoding="utf-8")
            self.assertIn("dist_single_4ch", summary_text)
            self.assertNotIn("dist_only_4ch", summary_text)

            self.assertTrue((temp_root / "reports" / "wake_field_summary.md").exists())
            self.assertTrue((temp_root / "reports" / "wake_field_reconstruction_summary.md").exists())

    def test_distance_crop_uses_physical_channel_height(self) -> None:
        box_ref = build_distance_crop_box(
            downstream_h=1.0,
            height_mode="full",
            physical_h=1.0,
            canvas_x_start=3.1,
            canvas_x_end=10.0,
            canvas_y_min=0.0,
            canvas_y_max=1.0,
        )
        box_padded = build_distance_crop_box(
            downstream_h=1.0,
            height_mode="full",
            physical_h=1.0,
            canvas_x_start=3.1,
            canvas_x_end=10.0,
            canvas_y_min=-0.02,
            canvas_y_max=1.02,
        )
        self.assertAlmostEqual(box_ref[0], box_padded[0], places=12)
        self.assertEqual(box_ref[1:], box_padded[1:])

    def test_lazy_resnet_entrypoints_do_not_import_vit_dependencies(self) -> None:
        code = textwrap.dedent(
            """
            import builtins

            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "timm" or name.startswith("timm"):
                    raise ModuleNotFoundError(f"blocked import: {name}")
                if name == "vision.mae_vit_model" or name.startswith("vision.mae_vit_model"):
                    raise ModuleNotFoundError(f"blocked import: {name}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            import ml.train_wake
            import ml.reconstruct_wake

            print("ok")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
