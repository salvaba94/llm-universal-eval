from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_universal_eval.catalog import LIGHTEVAL_BENCHMARKS
from llm_universal_eval.core import BenchmarkCommand
from llm_universal_eval.core import build_base_urls, get_required


def _write_lighteval_config(config: dict[str, Any], output_dir: Path) -> Path:
    model_name = get_required(config, "model.name")
    _, lighteval_url = build_base_urls(config)
    evaluation = config.get("evaluation", {})
    generation = evaluation.get("generation", {})
    lighteval = evaluation.get("lighteval", {})

    config_data = {
        "model_parameters": {
            "provider": lighteval.get("provider", "hosted_vllm"),
            "model_name": lighteval.get("model_name", f"hosted_vllm/{model_name}"),
            "base_url": lighteval_url,
            "api_key": lighteval.get("api_key", ""),
            "timeout": lighteval.get("timeout", 3600),
            "max_model_length": lighteval.get("max_model_length", 96000),
            "concurrent_requests": lighteval.get("concurrent_requests", 64),
            "generation_parameters": {
                "temperature": generation.get("temperature", 1.0),
                "max_new_tokens": generation.get("max_gen_toks", 64000),
                "top_p": generation.get("top_p", 0.95),
                "top_k": generation.get("top_k", 20),
                "min_p": generation.get("min_p", 0.0),
                "presence_penalty": generation.get("presence_penalty", 1.5),
                "repetition_penalty": generation.get("repetition_penalty", 1.0),
                "seed": generation.get("seed", 0),
            },
        }
    }

    config_path = output_dir / "litellm_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
    return config_path


def build_lighteval_commands(
    config: dict[str, Any], output_dir: Path
) -> list[BenchmarkCommand]:
    config_path = _write_lighteval_config(config, output_dir)
    commands: list[BenchmarkCommand] = []

    for benchmark in LIGHTEVAL_BENCHMARKS:
        benchmark_output_dir = output_dir / "lighteval" / benchmark["name"]
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        commands.append(
            BenchmarkCommand(
                name=benchmark["name"],
                harness="lighteval",
                command=[
                    "lighteval",
                    "endpoint",
                    "litellm",
                    str(config_path),
                    benchmark["task"],
                    "--output-dir",
                    str(benchmark_output_dir),
                    "--save-details",
                ],
            )
        )

    return commands