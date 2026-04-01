from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return data


def get_required(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing required config key: {path}")
        current = current[part]
    return current


def build_base_urls(config: dict[str, Any]) -> tuple[str, str]:
    host = get_required(config, "server.host")
    port = get_required(config, "server.port")
    root = f"http://{host}:{port}"
    return f"{root}/v1/chat/completions", f"{root}/v1"


def resolve_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config.get("evaluation", {}).get("output_dir", "results"))
    return (config_path.parent.parent / output_dir).resolve()