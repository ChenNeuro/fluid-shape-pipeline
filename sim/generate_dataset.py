from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from sim.case_builder import CaseSpec, build_case_specs
from sim.config import load_config, repo_root
from sim.data_schema import find_metadata_json
from sim.logging_utils import setup_logger
from sim.solvers import build_simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw probe datasets")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--solver", default=None, choices=["synthetic", "openfoam"], help="Override solver in config")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (0/1 for sequential)")
    return parser.parse_args()


def run_case_with_retries(case_spec: CaseSpec, simulator, max_retries: int, raw_root: Path, logger):
    start = time.time()
    attempts = 0
    last_error = ""

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            case_dir = raw_root / case_spec.case_id
            output_csv = simulator.run_case(case_spec, case_dir)
            metadata_json = find_metadata_json(case_dir)
            elapsed = time.time() - start
            logger.info("Case %s succeeded on attempt %d in %.2fs", case_spec.case_id, attempts, elapsed)
            return {
                **case_spec.to_dict(),
                "status": "success",
                "attempts": attempts,
                "error": "",
                "elapsed_s": round(elapsed, 4),
                "output_csv": str(output_csv),
                "metadata_json": str(metadata_json),
            }
        except Exception as exc:  # pylint: disable=broad-except
            last_error = str(exc)
            logger.warning("Case %s failed on attempt %d: %s", case_spec.case_id, attempts, last_error)

    elapsed = time.time() - start
    logger.error("Case %s failed after %d attempts", case_spec.case_id, attempts)
    return {
        **case_spec.to_dict(),
        "status": "failed",
        "attempts": attempts,
        "error": last_error,
        "elapsed_s": round(elapsed, 4),
        "output_csv": "",
        "metadata_json": "",
    }


def build_index_rows(manifest_df: pd.DataFrame, solver_name: str) -> list[dict]:
    rows: list[dict] = []
    for _, item in manifest_df.iterrows():
        case_id = str(item["case_id"])
        row = {
            "case_id": case_id,
            "solver": solver_name,
            "status": str(item["status"]),
            "shape": str(item["shape"]),
            "Re": int(item["re"]),
            "dy": float(item["dy"]),
            "eps": float(item["eps"]),
            "seed": int(item["seed"]),
            "attempts": int(item["attempts"]),
            "elapsed_s": float(item["elapsed_s"]),
            "error": str(item["error"]) if pd.notna(item["error"]) else "",
            "probes_csv": str(item["output_csv"]),
            "metadata_json": str(item["metadata_json"]),
        }

        if row["status"] == "success" and row["metadata_json"]:
            try:
                payload = json.loads(Path(row["metadata_json"]).read_text(encoding="utf-8"))
                probes = payload.get("probes", {})
                sampling = payload.get("sampling", {})
                row.update(
                    {
                        "n_probes": int(probes.get("count", 0)) if probes.get("count") is not None else 0,
                        "probe_x": probes.get("x"),
                        "probe_y_min": min(probes.get("y", [])) if probes.get("y") else None,
                        "probe_y_max": max(probes.get("y", [])) if probes.get("y") else None,
                        "dt": sampling.get("dt"),
                        "n_samples": int(sampling.get("n_samples", 0)) if sampling.get("n_samples") is not None else 0,
                        "t_start": sampling.get("t_start"),
                        "t_end": sampling.get("t_end"),
                    }
                )
            except Exception:  # pylint: disable=broad-except
                row["index_parse_error"] = "failed_to_parse_metadata"

        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = repo_root()
    logger = setup_logger("dataset", root / "logs" / "dataset.log")

    solver_name = args.solver or cfg["project"].get("solver", "synthetic")
    workers = args.workers if args.workers is not None else int(cfg["simulation"].get("workers", 0))
    max_retries = int(cfg["simulation"].get("max_retries", 1))

    raw_root = root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    case_specs = build_case_specs(cfg)
    simulator = build_simulator(solver_name, cfg, logger)

    logger.info("Starting dataset generation with solver=%s, workers=%d, cases=%d", solver_name, workers, len(case_specs))

    results = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_case_with_retries, case_spec, simulator, max_retries, raw_root, logger): case_spec
                for case_spec in case_specs
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for case_spec in case_specs:
            results.append(run_case_with_retries(case_spec, simulator, max_retries, raw_root, logger))

    manifest_df = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    manifest_path = raw_root / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    index_df = pd.DataFrame(build_index_rows(manifest_df, solver_name)).sort_values("case_id").reset_index(drop=True)
    index_path = raw_root / "index.csv"
    index_df.to_csv(index_path, index=False)

    n_total = int(manifest_df.shape[0])
    n_success = int((manifest_df["status"] == "success").sum())
    n_failed = n_total - n_success
    failure_rate = (n_failed / n_total) if n_total else 0.0

    summary = {
        "solver": solver_name,
        "total_cases": n_total,
        "success_cases": n_success,
        "failed_cases": n_failed,
        "failure_rate": failure_rate,
        "manifest": str(manifest_path),
        "index": str(index_path),
    }

    summary_path = raw_root / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Dataset generation complete. success=%d failed=%d failure_rate=%.2f%%", n_success, n_failed, failure_rate * 100.0)


if __name__ == "__main__":
    main()
