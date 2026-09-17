"""Small configuration helpers with explicit validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return payload


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"missing configuration section: {name}")
    return value
