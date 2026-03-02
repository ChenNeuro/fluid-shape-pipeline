from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_from_root(config_path or DEFAULT_CONFIG_PATH)
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
