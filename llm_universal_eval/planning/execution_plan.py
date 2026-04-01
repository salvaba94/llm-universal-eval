from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_universal_eval.builders.lighteval import build_lighteval_commands
from llm_universal_eval.builders.lm_eval import build_lm_eval_commands
from llm_universal_eval.builders.swebench import build_swebench_commands
from llm_universal_eval.core import BenchmarkCommand


def build_execution_plan(
    config: dict[str, Any], output_dir: Path, config_path: Path
) -> list[BenchmarkCommand]:
    commands: list[BenchmarkCommand] = []
    commands.extend(build_lm_eval_commands(config, output_dir))
    commands.extend(build_lighteval_commands(config, output_dir))
    commands.extend(build_swebench_commands(config, output_dir, config_path))
    return commands


def should_include_benchmark(
    benchmark: BenchmarkCommand,
    harness_selection: list[str],
    benchmark_selection: set[str] | None,
) -> bool:
    harness_allowed = "all" in harness_selection or benchmark.harness in harness_selection
    benchmark_allowed = benchmark_selection is None or benchmark.name in benchmark_selection
    return harness_allowed and benchmark_allowed