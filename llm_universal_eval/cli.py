from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from llm_universal_eval.catalog import BENCHMARK_CHOICES, HARNESS_CHOICES
from llm_universal_eval.core import load_config, resolve_output_dir, run_command
from llm_universal_eval.planning import build_execution_plan, should_include_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch lm-eval and lighteval benchmark runs from a YAML config."
    )
    parser.add_argument(
        "--config",
        default="config/evaluation_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=HARNESS_CHOICES,
        default=["all"],
        help="Limit execution to a subset of harnesses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_CHOICES,
        help="Run only the selected benchmark names.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()

    try:
        config = load_config(config_path)
        output_dir = resolve_output_dir(config, config_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_selection = set(args.benchmarks) if args.benchmarks else None
        execution_plan = [
            benchmark
            for benchmark in build_execution_plan(config, output_dir, config_path)
            if should_include_benchmark(benchmark, args.only, benchmark_selection)
        ]
        if not execution_plan:
            raise ValueError("No benchmarks selected for execution.")

        for index, benchmark in enumerate(execution_plan, start=1):
            print(
                f"[{index}/{len(execution_plan)}] "
                f"Running {benchmark.name} ({benchmark.harness})"
            )
            run_command(benchmark.command, args.dry_run, cwd=benchmark.cwd)
    except (KeyError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())