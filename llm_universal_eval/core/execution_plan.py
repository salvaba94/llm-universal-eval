from __future__ import annotations

from pathlib import Path

from llm_universal_eval.evaluation.lighteval import build_lighteval_commands
from llm_universal_eval.evaluation.lm_eval import build_lm_eval_commands
from llm_universal_eval.evaluation.swebench import build_swebench_commands
from llm_universal_eval.config.config import AppConfig
from llm_universal_eval.utils.types import BenchmarkCommand
from llm_universal_eval.utils.utils import task_to_name

_HARNESS_NAMES = {"all", "lm_eval", "lighteval", "swebench"}


def _is_lighteval_task(name: str) -> bool:
    """Heuristic: lighteval tasks use 'module:task' or 'task|shots' format."""
    return ":" in name or "|" in name


def build_execution_plan(
    config: AppConfig, output_dir: Path, config_path: Path
) -> list[BenchmarkCommand]:
    # Collect explicitly requested task names that aren't harness-level selectors
    explicit = [s for s in config.benchmark if s not in _HARNESS_NAMES]

    # Known task names derived from each harness's default list
    known_lighteval = {task_to_name(t) for t in config.lighteval.tasks}
    known_lm_eval = {task_to_name(t) for t in config.lm_eval.tasks}

    # Route unknown explicit tasks to the right harness based on format
    extra_lighteval: list[str] = []
    extra_lm_eval: list[str] = []
    for sel in explicit:
        if task_to_name(sel) in known_lighteval or task_to_name(sel) in known_lm_eval:
            continue  # already covered by the default list
        if _is_lighteval_task(sel):
            extra_lighteval.append(sel)
        else:
            extra_lm_eval.append(sel)

    commands: list[BenchmarkCommand] = []
    commands.extend(build_lm_eval_commands(config, output_dir, extra_lm_eval))
    commands.extend(build_lighteval_commands(config, output_dir, extra_lighteval))
    commands.extend(build_swebench_commands(config, output_dir, config_path))
    return commands


def should_include_benchmark(
    benchmark: BenchmarkCommand,
    select: list[str],
) -> bool:
    if "all" in select:
        return True
    return any(
        benchmark.harness == s or benchmark.name == s or benchmark.name == task_to_name(s)
        for s in select
    )