from __future__ import annotations

import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from llm_universal_eval.config.config import AppConfig


def task_to_name(task: str) -> str:
    """Convert a harness task string to a filesystem-safe name.

    Strips the lighteval fewshot suffix (e.g. ``|0``) before sanitizing,
    so ``gpqa:diamond|0`` becomes ``gpqa_diamond`` rather than ``gpqa_diamond_0``.
    """
    name = re.sub(r"\|\d+$", "", task)  # strip trailing |<number>
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


def load_config(config_path: Path, overrides: list[str] | None = None) -> AppConfig:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = compose(config_name=config_path.stem, overrides=overrides)
    return AppConfig.from_omegaconf(cfg)


def build_base_urls(config: AppConfig) -> tuple[str, str]:
    host = config.model.host
    port = config.model.port
    root = f"http://{host}:{port}"
    return f"{root}/v1/chat/completions", f"{root}/v1"


def resolve_output_dir(config: AppConfig, base_dir: Path) -> Path:
    output_dir = Path(config.output_dir)
    if output_dir.is_absolute():
        return output_dir
    return (base_dir / output_dir).resolve()
