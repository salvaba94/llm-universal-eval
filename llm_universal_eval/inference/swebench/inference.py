"""SWE-bench inference against an OpenAI-compatible API endpoint."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "You will be given a GitHub repository issue. "
    "Produce a minimal unified diff patch (git diff format) that resolves the issue. "
    "Output ONLY the raw diff, with no explanations and no markdown code fences, "
    "starting with 'diff --git'."
)

_PROMPT_TEMPLATE = """\
Repository: {repo}

<issue>
{problem_statement}
</issue>

Provide the minimal git diff patch that fixes this issue."""


def _extract_diff(text: str) -> str:
    try:
        from swebench.inference.make_datasets.utils import extract_diff  # type: ignore

        result = extract_diff(text)
        if result:
            return result
    except Exception:
        pass

    for pattern in (r"```diff\n(.*?)```", r"```patch\n(.*?)```", r"```\n(.*?)```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("diff ") or "--- a/" in candidate:
                return candidate

    for line in text.splitlines():
        if line.startswith("diff --git") or line.startswith("--- a/"):
            return text[text.index(line) :]

    return text


def _build_user_message(instance: dict[str, Any]) -> str:
    text = instance.get("text")
    if text:
        return text
    return _PROMPT_TEMPLATE.format(
        repo=instance.get("repo", "unknown"),
        problem_statement=instance.get("problem_statement", ""),
    )


def run_inference(
    config: dict[str, Any],
    dataset_name: str,
    split: str,
    output_file: Path,
    shard_id: int | None,
    num_shards: int | None,
) -> None:
    from datasets import load_dataset  # type: ignore

    model_name = config["model"]["name"]
    host = config["server"]["host"]
    port = config["server"]["port"]
    base_url = f"http://{host}:{port}/v1"

    evaluation = config.get("evaluation", {})
    generation = evaluation.get("generation", {})
    swebench_cfg = evaluation.get("swebench", {})
    api_key = swebench_cfg.get("api_key", "EMPTY")

    client = OpenAI(api_key=api_key, base_url=base_url)

    logger.info("Loading dataset %s [split=%s]", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)

    existing_ids: set[str] = set()
    if output_file.exists():
        for raw in output_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                existing_ids.add(json.loads(raw)["instance_id"])

    temperature = generation.get("temperature", 1.0)
    top_p = generation.get("top_p", 0.95)
    max_tokens = generation.get("max_gen_toks", 64000)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    with output_file.open("a", encoding="utf-8") as handle:
        for index, instance in enumerate(dataset):
            instance_id = instance["instance_id"]
            if instance_id in existing_ids:
                continue

            logger.info("[%d/%d] Generating patch for %s", index + 1, total, instance_id)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(instance)},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            completion = response.choices[0].message.content or ""
            record = {
                "instance_id": instance_id,
                "model_patch": _extract_diff(completion),
                "model_name_or_path": model_name,
            }
            print(json.dumps(record), file=handle, flush=True)

    logger.info("Inference complete. Predictions written to %s", output_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench inference against an OpenAI-compatible endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML.")
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help=(
            "HuggingFace dataset name (must contain 'instance_id' and either "
            "'text' or 'problem_statement' columns)."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_file", required=True, help="Output predictions JSONL path.")
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_inference(
            config=config,
            dataset_name=args.dataset_name,
            split=args.split,
            output_file=Path(args.output_file),
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())