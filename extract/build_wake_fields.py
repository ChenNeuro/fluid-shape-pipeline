from __future__ import annotations

import argparse
import json

import pandas as pd

from sim.config import load_config
from sim.data_schema import find_wake_frames_npz
from sim.experiment import experiment_paths, write_run_manifest
from sim.logging_utils import setup_logger
from vision.wake_field_builder import build_case_wake_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build wake-field tensors from short wake frame sequences"
    )
    parser.add_argument(
        "--config", default="configs/wake_field_450.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment output directory. Defaults to runs/<config-name>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = experiment_paths(cfg, config_path=args.config, run_dir=args.run_dir)
    write_run_manifest(paths=paths, cfg=cfg, config_path=args.config, stage="wake_fields")
    logger = setup_logger("wake_fields", paths.logs_dir / "wake_fields.log")

    manifest_path = paths.raw_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run dataset generation first."
        )

    manifest = pd.read_csv(manifest_path)
    ok_cases = manifest[manifest["status"] == "success"].copy()
    if ok_cases.empty:
        raise RuntimeError("No successful cases found in manifest; cannot build wake fields")

    rows = []
    skipped = 0
    for _, item in ok_cases.iterrows():
        case_id = str(item["case_id"])
        case_dir = paths.raw_dir / case_id
        try:
            find_wake_frames_npz(case_dir)
        except FileNotFoundError:
            logger.warning(
                "Skipping case %s because wake_frames.npz is missing in %s", case_id, case_dir
            )
            skipped += 1
            continue

        try:
            rows.append(build_case_wake_field(case_dir, cfg))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Wake-field build failed for %s: %s", case_id, exc)
            skipped += 1

    if not rows:
        raise RuntimeError("Wake-field build produced no rows")

    output_dir = paths.wake_fields_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    pd.DataFrame(rows).sort_values("case_id").to_csv(index_path, index=False)

    summary_path = output_dir / "summary.json"
    summary = {
        "rows": len(rows),
        "skipped": skipped,
        "index_csv": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "Wake-field build complete. rows=%d skipped=%d -> %s", len(rows), skipped, index_path
    )


if __name__ == "__main__":
    main()
