from __future__ import annotations

import argparse

import pandas as pd

from extract.feature_engineering import extract_features_from_df
from sim.config import load_config
from sim.data_schema import find_probes_csv
from sim.experiment import experiment_paths, write_run_manifest
from sim.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length features from raw probe signals"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
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
    write_run_manifest(paths=paths, cfg=cfg, config_path=args.config, stage="features")

    logger = setup_logger("features", paths.logs_dir / "features.log")

    manifest_path = paths.raw_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run dataset generation first."
        )

    manifest = pd.read_csv(manifest_path)
    ok_cases = manifest[manifest["status"] == "success"].copy()
    if ok_cases.empty:
        raise RuntimeError("No successful cases found in manifest; cannot extract features")

    feat_cfg = cfg["features"]
    n_bands = int(feat_cfg["fft_bands"])
    pod_modes = int(feat_cfg["pod_modes"])
    add_pod = bool(feat_cfg.get("add_pod", True))

    rows = []
    for _, item in ok_cases.iterrows():
        case_id = item["case_id"]
        case_dir = paths.raw_dir / case_id
        try:
            raw_csv = find_probes_csv(case_dir)
        except FileNotFoundError:
            logger.warning("Skipping case %s because raw csv is missing in %s", case_id, case_dir)
            continue

        df = pd.read_csv(raw_csv)
        try:
            feature_values = extract_features_from_df(
                df, n_bands=n_bands, pod_modes=pod_modes, add_pod=add_pod
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Feature extraction failed for %s: %s", case_id, exc)
            continue

        row = {
            "case_id": case_id,
            "shape": item["shape"],
            "Re": int(item["re"]),
            "dy": float(item["dy"]),
            "eps": float(item["eps"]),
            "seed": int(item["seed"]),
        }
        row.update(feature_values)
        rows.append(row)

    if not rows:
        raise RuntimeError("Feature extraction produced no rows")

    features_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    feat_dir = paths.features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)
    output_csv = feat_dir / "features.csv"
    features_df.to_csv(output_csv, index=False)

    logger.info(
        "Feature extraction complete. rows=%d cols=%d -> %s",
        features_df.shape[0],
        features_df.shape[1],
        output_csv,
    )


if __name__ == "__main__":
    main()
