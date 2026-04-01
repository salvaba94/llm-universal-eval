from __future__ import annotations

from typing import Any


def serialize_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_key_value_string(items: list[tuple[str, Any]]) -> str:
    return ",".join(f"{key}={serialize_value(value)}" for key, value in items)