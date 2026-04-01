from __future__ import annotations

from pathlib import Path

from llm_universal_eval.evaluation.common import format_key_value_string
from llm_universal_eval.config.config import AppConfig
from llm_universal_eval.utils.types import BenchmarkCommand
from llm_universal_eval.utils.utils import build_base_urls, task_to_name


def build_lm_eval_commands(
    config: AppConfig, output_dir: Path, extra_tasks: list[str] | None = None
) -> list[BenchmarkCommand]:
    model_name = config.model.name
    chat_completions_url, _ = build_base_urls(config)
    generation = config.model
    lm_eval = config.lm_eval
    seed = config.seed

    gen_kwargs = format_key_value_string(
        [
            ("do_sample", generation.do_sample),
            ("temperature", generation.temperature),
            ("top_p", generation.top_p),
            ("top_k", generation.top_k),
            ("min_p", generation.min_p),
            ("max_gen_toks", generation.max_gen_toks),
            ("presence_penalty", generation.presence_penalty),
            ("repetition_penalty", generation.repetition_penalty),
            ("seed", seed),
        ]
    )

    tasks = list(lm_eval.tasks)
    for task in extra_tasks or []:
        if task not in tasks:
            tasks.append(task)

    commands: list[BenchmarkCommand] = []

    for task in tasks:
        name = task_to_name(task)
        benchmark_output_dir = output_dir / "lm_eval" / name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        model_args = format_key_value_string(
            [
                ("model", model_name),
                ("max_length", lm_eval.max_length),
                ("base_url", chat_completions_url),
                ("num_concurrent", lm_eval.num_concurrent),
                ("max_retries", lm_eval.max_retries),
                ("tokenized_requests", lm_eval.tokenized_requests),
                ("tokenizer_backend", lm_eval.tokenizer_backend),
            ]
        )

        command = [
            "lm_eval",
            "--model",
            "local-chat-completions",
            "--tasks",
            task,
            "--model_args",
            model_args,
            "--apply_chat_template",
            "--output_path",
            str(benchmark_output_dir / "results.json"),
            "--seed",
            str(seed),
            "--gen_kwargs",
            gen_kwargs,
        ]
        commands.append(
            BenchmarkCommand(
                name=name,
                harness="lm_eval",
                command=command,
            )
        )

    return commands