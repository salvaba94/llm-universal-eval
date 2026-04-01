from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCommand:
    name: str
    harness: str
    command: list[str]
    cwd: Path | None = None