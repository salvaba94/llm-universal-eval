from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_universal_eval.builders.common import format_key_value_string
from llm_universal_eval.catalog import LM_EVAL_BENCHMARKS
from llm_universal_eval.core import BenchmarkCommand
from llm_universal_eval.core import build_base_urls, get_required


def build_lm_eval_commands(
    config: dict[str, Any], output_dir: Path
) -> list[BenchmarkCommand]:
    model_name = get_required(config, "model.name")
    chat_completions_url, _ = build_base_urls(config)
    evaluation = config.get("evaluation", {})
    generation = evaluation.get("generation", {})
    lm_eval = evaluation.get("lm_eval", {})
    seed = evaluation.get("seed", 42)

    gen_kwargs = format_key_value_string(
        [
            ("do_sample", generation.get("do_sample", True)),
            ("temperature", generation.get("temperature", 1.0)),
            ("top_p", generation.get("top_p", 0.95)),
            ("top_k", generation.get("top_k", 20)),
            ("min_p", generation.get("min_p", 0.0)),
            ("max_gen_toks", generation.get("max_gen_toks", 64000)),
            ("presence_penalty", generation.get("presence_penalty", 1.5)),
            ("repetition_penalty", generation.get("repetition_penalty", 1.0)),
            ("seed", seed),
        ]
    )

    task_timeouts = lm_eval.get("task_timeouts", {})
    commands: list[BenchmarkCommand] = []

    for benchmark in LM_EVAL_BENCHMARKS:
        benchmark_output_dir = output_dir / "lm_eval" / benchmark["name"]
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        model_args = format_key_value_string(
            [
                ("model", model_name),
                ("max_length", lm_eval.get("max_length", 96000)),
                ("base_url", chat_completions_url),
                ("num_concurrent", lm_eval.get("num_concurrent", 128)),
                ("max_retries", lm_eval.get("max_retries", 3)),
                ("tokenized_requests", lm_eval.get("tokenized_requests", False)),
                ("tokenizer_backend", lm_eval.get("tokenizer_backend", None)),
                (
                    "timeout",
                    task_timeouts.get(benchmark["task"], benchmark["timeout"]),
                ),
            ]
        )

        command = [
            "lm_eval",
            "--model",
            "local-chat-completions",
            "--tasks",
            benchmark["task"],
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
        if benchmark["num_fewshot"] is not None:
            command.extend(["--num_fewshot", str(benchmark["num_fewshot"])])

        commands.append(
            BenchmarkCommand(
                name=benchmark["name"],
                harness="lm_eval",
                command=command,
            )
        )

    return commands