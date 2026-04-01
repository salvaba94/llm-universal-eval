from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from llm_universal_eval.core import BenchmarkCommand


def build_swebench_commands(
    config: dict[str, Any], output_dir: Path, config_path: Path
) -> list[BenchmarkCommand]:
    swebench_cfg = config.get("evaluation", {}).get("swebench", {})
    dataset_name = swebench_cfg.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
    split = swebench_cfg.get("split", "test")
    max_workers = swebench_cfg.get("max_workers", 4)
    run_id = swebench_cfg.get("run_id", "eval")

    benchmark_output_dir = output_dir / "swebench"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_short = dataset_name.split("/")[-1]
    predictions_file = benchmark_output_dir / f"{dataset_short}.jsonl"

    inference_cmd = [
        sys.executable,
        "-m",
        "llm_universal_eval.frameworks.swebench.inference",
        "--config",
        str(config_path),
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--output_file",
        str(predictions_file),
    ]

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
            name="swebench_inference",
            harness="swebench",
            command=inference_cmd,
        ),
        BenchmarkCommand(
            name="swebench_evaluation",
            harness="swebench",
            command=eval_cmd,
            cwd=benchmark_output_dir,
        ),
    ]