from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCommand:  # noqa: F811
    name: str
    harness: str
    command: list[str]
    step: str | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None
