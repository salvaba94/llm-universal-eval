from __future__ import annotations

import sys
from pathlib import Path

from llm_universal_eval.config.config import AppConfig
from llm_universal_eval.utils.types import BenchmarkCommand
from llm_universal_eval.utils.utils import build_base_urls


def build_swebench_commands(
    config: AppConfig, output_dir: Path, config_path: Path
) -> list[BenchmarkCommand]:
    swebench_cfg = config.swebench
    dataset_name = swebench_cfg.dataset_name
    split = swebench_cfg.split
    max_workers = swebench_cfg.max_workers
    run_id = swebench_cfg.run_id

    _, base_url = build_base_urls(config)
    api_key = swebench_cfg.api_key
    model_name = config.model.name
    generation = config.model

    benchmark_output_dir = output_dir / "swebench"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    # If a previous inference file is provided, skip inference entirely
    if swebench_cfg.predictions_path:
        predictions_file = Path(swebench_cfg.predictions_path)
        eval_cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--predictions_path",
            str(predictions_file),
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
        ]
        return [
            BenchmarkCommand(
                name="swebench",
                harness="swebench",
                step="evaluation",
                command=eval_cmd,
                cwd=benchmark_output_dir,
            )
        ]

    # run_api.py requires a pre-processed text dataset (with a 'text' column).
    # We create it first via make_datasets.create_text_dataset, then pass the
    # local path to run_api so it can sort by length and resume on restart.
    text_dataset_dir = benchmark_output_dir / "text_dataset"
    model_nickname = Path(model_name).name
    dataset_short = dataset_name.split("/")[-1]

    make_dataset_cmd = [
        sys.executable,
        "-m",
        "swebench.inference.make_datasets.create_text_dataset",
        "--dataset_name_or_path",
        dataset_name,
        "--output_dir",
        str(text_dataset_dir),
        "--prompt_style",
        swebench_cfg.prompt_style,
        "--splits",
        split,
    ]

    model_args = (
        f"temperature={generation.temperature}"
        f",top_p={generation.top_p}"
        f",max_tokens={generation.max_gen_toks}"
    )

    # create_text_dataset writes a subdirectory named:
    #   {dataset_short}__{prompt_style}__fs-{file_source}
    # where file_source defaults to "oracle".  run_api then loads it with
    # load_from_disk() and accesses dataset[split].
    text_dataset_path = text_dataset_dir / f"{dataset_short}__{swebench_cfg.prompt_style}__fs-oracle"
    # run_api names its output file using the last component of dataset_name_or_path
    predictions_file = benchmark_output_dir / f"{model_nickname}__{text_dataset_path.name}__{split}.jsonl"

    inference_cmd = [
        sys.executable,
        "-m",
        "llm_universal_eval.inference.run_api",
        "--dataset_name_or_path",
        str(text_dataset_path),
        "--split",
        split,
        "--model_name_or_path",
        model_name,
        "--base_url",
        base_url,
        "--output_dir",
        str(benchmark_output_dir),
        "--model_args",
        model_args,
    ]

    inference_env = {
        "OPENAI_API_KEY": api_key,
    }

    eval_cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_file),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]

    return [
        BenchmarkCommand(
            name="swebench",
            harness="swebench",
            step="make_dataset",
            command=make_dataset_cmd,
        ),
        BenchmarkCommand(
            name="swebench",
            harness="swebench",
            step="inference",
            command=inference_cmd,
            env=inference_env,
        ),
        BenchmarkCommand(
            name="swebench",
            harness="swebench",
            step="evaluation",
            command=eval_cmd,
            cwd=benchmark_output_dir,
        ),
    ]