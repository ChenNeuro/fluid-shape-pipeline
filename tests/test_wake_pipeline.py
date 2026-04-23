from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "wake_field_smoke.yaml"


class WakePipelineSmokeTest(unittest.TestCase):
    def test_end_to_end_smoke_pipeline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wake-pipeline-smoke-") as tmpdir:
            temp_root = Path(tmpdir)
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

            field_payload = np.load(field_path)
            self.assertEqual(int(field_payload["field_raw"].shape[0]), 4)  # 4 channels
            self.assertEqual(int(field_payload["crops"].shape[0]), 3)  # 3 scales: dist1.0_full/half/quarter
            self.assertEqual(int(field_payload["crops"].shape[1]), 4)  # 4 channels

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("field_channels", metadata)
            self.assertIn("crop_boxes", metadata)
            hotspot_box = metadata["crop_boxes"]["hotspot"]
            self.assertGreaterEqual(hotspot_box[0], 0.0)
            self.assertGreaterEqual(hotspot_box[1], 0.0)
            self.assertLessEqual(hotspot_box[2], 1.0)
            self.assertLessEqual(hotspot_box[3], 1.0)

            self.assertTrue((temp_root / "models" / "wake_field_main.pt").exists())
            self.assertTrue((temp_root / "reports" / "wake_field_summary.md").exists())
            self.assertTrue((temp_root / "reports" / "wake_field_reconstruction_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
