from __future__ import annotations

import json
from pathlib import Path


PROBES_FILENAME = "probes.csv"
LEGACY_PROBES_FILENAME = "probe_u.csv"
METADATA_FILENAME = "metadata.json"
LEGACY_METADATA_FILENAME = "meta.json"


def find_probes_csv(case_dir: Path) -> Path:
    primary = case_dir / PROBES_FILENAME
    if primary.exists():
        return primary

    legacy = case_dir / LEGACY_PROBES_FILENAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(f"No probe csv found in {case_dir}. Expected {PROBES_FILENAME} or {LEGACY_PROBES_FILENAME}.")


def find_metadata_json(case_dir: Path) -> Path:
    primary = case_dir / METADATA_FILENAME
    if primary.exists():
        return primary

    legacy = case_dir / LEGACY_METADATA_FILENAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(f"No metadata file found in {case_dir}. Expected {METADATA_FILENAME} or {LEGACY_METADATA_FILENAME}.")


def write_metadata(case_dir: Path, payload: dict) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / METADATA_FILENAME
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output

