from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _print_command(command: list[str]) -> None:
    print("+", shlex.join(command))


def run_command(command: list[str], dry_run: bool, cwd: Path | None = None) -> None:
    _print_command(command)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=cwd)