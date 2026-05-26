from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.config import repo_root, resolve_from_root

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "timm",
    "pytest",
)


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_id: str
    run_dir: Path
    data_dir: Path
    raw_dir: Path
    features_dir: Path
    wake_fields_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path


def infer_experiment_id(cfg: dict[str, Any], config_path: str | Path | None) -> str:
    project_cfg = cfg.get("project", {})
    configured = project_cfg.get("experiment_id") or project_cfg.get("name")
    if configured:
        return str(configured)
    if config_path:
        return Path(config_path).stem
    return "default"


def experiment_paths(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> ExperimentPaths:
    experiment_id = infer_experiment_id(cfg, config_path)
    output_cfg = cfg.get("outputs", {})
    configured_run_dir = run_dir or output_cfg.get("run_dir")
    resolved_run_dir = (
        resolve_from_root(configured_run_dir)
        if configured_run_dir
        else repo_root() / "runs" / experiment_id
    )

    data_dir = resolved_run_dir / "data"
    return ExperimentPaths(
        experiment_id=experiment_id,
        run_dir=resolved_run_dir,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        features_dir=data_dir / "features",
        wake_fields_dir=data_dir / "wake_fields",
        models_dir=resolved_run_dir / "models",
        reports_dir=resolved_run_dir / "reports",
        logs_dir=resolved_run_dir / "logs",
    )


def ensure_experiment_dirs(paths: ExperimentPaths) -> None:
    for path in (
        paths.raw_dir,
        paths.features_dir,
        paths.wake_fields_dir,
        paths.models_dir,
        paths.reports_dir,
        paths.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def git_metadata() -> dict[str, Any]:
    diff = _git_value(["diff", "--"])
    staged_diff = _git_value(["diff", "--cached", "--"])
    status = _git_value(["status", "--short", "--branch"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "status": status,
        "dirty": bool(
            status and any(line and not line.startswith("##") for line in status.splitlines())
        ),
        "diff_sha256": _sha256_text(diff or ""),
        "staged_diff_sha256": _sha256_text(staged_diff or ""),
    }


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
    }


def write_requirements_freeze(paths: ExperimentPaths) -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return
    (paths.run_dir / "requirements-freeze.txt").write_text(completed.stdout, encoding="utf-8")


def write_run_manifest(
    *,
    paths: ExperimentPaths,
    cfg: dict[str, Any],
    config_path: str | Path | None,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    ensure_experiment_dirs(paths)
    payload = {
        "experiment_id": paths.experiment_id,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(paths.run_dir),
        "config_path": str(resolve_from_root(config_path)) if config_path else None,
        "config_snapshot": cfg,
        "config_sha256": _sha256_text(json.dumps(cfg, sort_keys=True, default=str)),
        "git": git_metadata(),
        "environment": environment_metadata(),
    }
    if extra:
        payload["extra"] = extra

    (paths.run_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with (paths.run_dir / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    write_requirements_freeze(paths)
