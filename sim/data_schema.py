from __future__ import annotations

import json
from pathlib import Path

PROBES_FILENAME = "probes.csv"
LEGACY_PROBES_FILENAME = "probe_u.csv"
METADATA_FILENAME = "metadata.json"
LEGACY_METADATA_FILENAME = "meta.json"
WAKE_FRAMES_FILENAME = "wake_frames.npz"
WAKE_FIELD_FILENAME = "wake_field.npz"


def find_probes_csv(case_dir: Path) -> Path:
    primary = case_dir / PROBES_FILENAME
    if primary.exists():
        return primary

    legacy = case_dir / LEGACY_PROBES_FILENAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"No probe csv found in {case_dir}. Expected {PROBES_FILENAME} or {LEGACY_PROBES_FILENAME}."
    )


def find_metadata_json(case_dir: Path) -> Path:
    primary = case_dir / METADATA_FILENAME
    if primary.exists():
        return primary

    legacy = case_dir / LEGACY_METADATA_FILENAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"No metadata file found in {case_dir}. Expected {METADATA_FILENAME} or {LEGACY_METADATA_FILENAME}."
    )


def write_metadata(case_dir: Path, payload: dict) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / METADATA_FILENAME
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def read_metadata(case_dir: Path) -> dict:
    path = find_metadata_json(case_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def find_wake_frames_npz(case_dir: Path) -> Path:
    path = case_dir / WAKE_FRAMES_FILENAME
    if path.exists():
        return path
    raise FileNotFoundError(
        f"No wake frame artifact found in {case_dir}. Expected {WAKE_FRAMES_FILENAME}."
    )


def find_wake_field_npz(case_dir: Path) -> Path:
    path = case_dir / WAKE_FIELD_FILENAME
    if path.exists():
        return path
    raise FileNotFoundError(
        f"No wake field artifact found in {case_dir}. Expected {WAKE_FIELD_FILENAME}."
    )
