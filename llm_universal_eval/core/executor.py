from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def _print_command(command: list[str], env: dict[str, str] | None = None) -> None:
    if env:
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        print("+", env_str, shlex.join(command))
    else:
        print("+", shlex.join(command))


def run_command(
    command: list[str],
    dry_run: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    merged_env = {**os.environ, **(env or {})}
    _print_command(command, env)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=cwd, env=merged_env)