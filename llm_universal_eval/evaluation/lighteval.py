from __future__ import annotations

from pathlib import Path

import yaml

from llm_universal_eval.config.config import AppConfig
from llm_universal_eval.utils.types import BenchmarkCommand
from llm_universal_eval.utils.utils import build_base_urls, task_to_name


def _write_lighteval_config(config: AppConfig, output_dir: Path) -> Path:
    model_name = config.model.name
    _, lighteval_url = build_base_urls(config)
    generation = config.model
    lighteval = config.lighteval

    config_data = {
        "model_parameters": {
            "provider": lighteval.provider,
            "model_name": lighteval.model_name or f"hosted_vllm/{model_name}",
            "base_url": lighteval_url,
            "api_key": lighteval.api_key,
            "timeout": lighteval.timeout,
            "concurrent_requests": lighteval.concurrent_requests,
            "generation_parameters": {
                "temperature": generation.temperature,
                "max_new_tokens": generation.max_gen_toks,
                "top_p": generation.top_p,
                "top_k": generation.top_k,
                "min_p": generation.min_p,
                "presence_penalty": generation.presence_penalty,
                "repetition_penalty": generation.repetition_penalty,
                "seed": config.seed,
            },
        }
    }

    config_path = output_dir / "litellm_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
    return config_path


def build_lighteval_commands(
    config: AppConfig, output_dir: Path, extra_tasks: list[str] | None = None
) -> list[BenchmarkCommand]:
    config_path = _write_lighteval_config(config, output_dir)
    commands: list[BenchmarkCommand] = []

    tasks = list(config.lighteval.tasks)
    for task in extra_tasks or []:
        # ensure the lighteval |N fewshot suffix is present
        if "|" not in task:
            task = f"{task}|0"
        if task not in tasks:
            tasks.append(task)

    for task in tasks:
        name = task_to_name(task)
        benchmark_output_dir = output_dir / "lighteval" / name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        commands.append(
            BenchmarkCommand(
                name=name,
                harness="lighteval",
                command=[
                    "lighteval",
                    "endpoint",
                    "litellm",
                    str(config_path),
                    task,
                    "--output-dir",
                    str(benchmark_output_dir),
                    "--save-details",
                ],
            )
        )

    return commands