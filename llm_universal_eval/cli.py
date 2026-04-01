from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from llm_universal_eval.core import AppConfig, resolve_output_dir, run_command
from llm_universal_eval.core import build_execution_plan, should_include_benchmark


@hydra_main(
    version_base=None,
    config_path="pkg://llm_universal_eval.config",
    config_name="config",
)
def main(cfg: DictConfig) -> int:
    try:
        config = AppConfig.from_omegaconf(cfg)

        base_dir = Path.cwd().resolve()
        output_dir = resolve_output_dir(config, base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_config_path = output_dir / "config.yaml"
        OmegaConf.save(cfg, resolved_config_path)

        execution_plan = [
            benchmark
            for benchmark in build_execution_plan(
                config,
                output_dir,
                resolved_config_path,
            )
            if should_include_benchmark(benchmark, config.benchmark)
        ]
        if not execution_plan:
            raise ValueError("No benchmarks selected for execution.")

        for index, benchmark in enumerate(execution_plan, start=1):
            label = f"{benchmark.name}/{benchmark.step}" if benchmark.step else benchmark.name
            print(f"[{index}/{len(execution_plan)}] Running {label} ({benchmark.harness})")
            run_command(benchmark.command, config.dry_run, cwd=benchmark.cwd, env=benchmark.env)
    except (KeyError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())