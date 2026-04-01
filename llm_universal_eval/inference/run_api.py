#!/usr/bin/env python3
"""Run SWE-bench inference against any OpenAI-compatible endpoint (e.g. vLLM).

This is a drop-in replacement for swebench.inference.run_api that:
- Accepts ``--base_url`` to point at a local vLLM / LiteLLM proxy
- Removes the ``choices`` restriction on ``--model_name_or_path``
- Uses ``cl100k_base`` tiktoken encoding as a fallback for unknown models
- Skips cost tracking (irrelevant for local inference)
- Produces exactly the same output JSONL format as the original so that
  ``swebench.harness.run_evaluation`` can consume the results directly.
"""

from __future__ import annotations

import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk
from openai import OpenAI, BadRequestError
from swebench.inference.make_datasets.utils import extract_diff
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _tokenize(text: str, encoding) -> int:
    return len(encoding.encode(text))


def _get_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(5))
def _call_chat(client: OpenAI, model_name: str, text: str, temperature: float, top_p: float, **model_args) -> str:
    system_message, user_message = text.split("\n", 1)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
        **model_args,
    )
    return response.choices[0].message.content


def openai_compatible_inference(
    test_dataset,
    model_name_or_path: str,
    output_file: Path,
    model_args: dict,
    existing_ids: set,
    max_context_len: int,
    client: OpenAI,
):
    encoding = _get_encoding(model_name_or_path)
    test_dataset = test_dataset.filter(
        lambda x: _tokenize(x["text"], encoding) <= max_context_len,
        desc="Filtering by context length",
        load_from_cache_file=False,
    )

    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    logger.info(f"temperature={temperature}, top_p={top_p}")
    logger.info(f"Running inference on {len(test_dataset)} instances")

    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "text": f"{datum['text']}\n\n",
            }
            try:
                completion = _call_chat(
                    client,
                    model_name_or_path,
                    output_dict["text"],
                    temperature,
                    top_p,
                    **model_args,
                )
            except BadRequestError as e:
                logger.error(f"BadRequestError for {instance_id}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {instance_id}: {e}")
                continue
            output_dict["full_output"] = completion
            output_dict["model_patch"] = extract_diff(completion)
            print(json.dumps(output_dict), file=f, flush=True)


def parse_model_args(model_args: str | None) -> dict:
    kwargs: dict = {}
    if model_args is None:
        return kwargs
    for arg in model_args.split(","):
        key, value = arg.split("=")
        if value in {"True", "False"}:
            kwargs[key] = value == "True"
        elif value.isnumeric():
            kwargs[key] = int(value)
        elif value.replace(".", "", 1).isnumeric():
            kwargs[key] = float(value)
        elif value in {"None"}:
            kwargs[key] = None
        elif (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            kwargs[key] = value[1:-1]
        else:
            kwargs[key] = value
    return kwargs


def main(
    dataset_name_or_path: str,
    split: str,
    model_name_or_path: str,
    output_dir: str,
    base_url: str | None,
    model_args: str | None,
    max_context_len: int,
    shard_id: int | None,
    num_shards: int | None,
):
    if shard_id is None and num_shards is not None:
        logger.warning(f"num_shards={num_shards} but shard_id is None, ignoring")
    if shard_id is not None and num_shards is None:
        logger.warning(f"shard_id={shard_id} but num_shards is None, ignoring")

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=resolved_base_url)

    parsed_model_args = parse_model_args(model_args)

    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name

    output_file = f"{model_nickname}__{Path(dataset_name_or_path).name}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir) / (output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data["instance_id"])
    logger.info(f"Found {len(existing_ids)} already-completed ids")

    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset {dataset_name_or_path}")
    dataset = dataset[split]

    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))

    if existing_ids:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )

    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)

    openai_compatible_inference(
        test_dataset=dataset,
        model_name_or_path=model_name_or_path,
        output_file=output_file,
        model_args=parsed_model_args,
        existing_ids=existing_ids,
        max_context_len=max_context_len,
        client=client,
    )
    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path (output of create_text_dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name sent to the API (e.g. Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the predictions JSONL file",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1). "
             "Falls back to $OPENAI_BASE_URL if not set.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="Comma-separated model arguments, e.g. 'temperature=0.2,top_p=0.95,max_tokens=4096'",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=128_000,
        help="Maximum number of tokens per instance (instances exceeding this are skipped)",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard index to process (for parallelism)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Total number of shards (for parallelism)",
    )
    args = parser.parse_args()
    main(**vars(args))
