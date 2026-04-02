# LLM Universal Eval

A unified launcher for running [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/neuralmagic/lighteval), and [SWE-bench](https://github.com/SWE-bench/SWE-bench) against a vLLM-served model.

## Setup

```bash
uv sync
uv pip install -e .
```

The package installs a console script named `llm-universal-eval`.

## Configuration

All configuration lives under `llm_universal_eval/config/`. The top-level `config.yaml` pulls in four sub-configs via Hydra defaults:

```
config.yaml          # top-level: output_dir, seed, benchmark, dry_run
model/default.yaml   # model name, host, port, generation parameters
lm_eval/default.yaml # lm-eval harness settings + task list
lighteval/default.yaml # lighteval harness settings + task list
swebench/default.yaml  # SWE-bench dataset, split, workers, predictions_path
```

### Model (`model/default.yaml`)

| Key | Description |
|---|---|
| `name` | Model name as registered in vLLM |
| `host` / `port` | vLLM server address |
| `temperature`, `top_p`, `top_k`, `min_p` | Sampling parameters |
| `max_gen_toks` | Maximum output tokens |
| `presence_penalty`, `repetition_penalty` | Penalty parameters |

### lm-eval (`lm_eval/default.yaml`)

| Key | Description |
|---|---|
| `tasks` | List of lm-eval task names to run |
| `max_length` | Maximum input length |
| `num_concurrent` | Concurrent requests to vLLM |
| `max_retries` | Retries on request failure |
| `tokenized_requests` | Whether to send tokenized inputs |
| `tokenizer_backend` | Tokenizer backend override |

Default tasks: `gsm8k_platinum_cot_llama`, `ifeval`, `mmlu_pro`. These tasks will be launchjed if the user invokes lm-eval evaluation battery.

### lighteval (`lighteval/default.yaml`)

| Key | Description |
|---|---|
| `tasks` | List of lighteval task strings (e.g. `math_500\`) |
| `provider` | LiteLLM provider (default: `hosted_vllm`) |
| `model_name` | Override model name passed to LiteLLM |
| `api_key` | API key (empty for local vLLM) |
| `timeout` | Request timeout in seconds |
| `concurrent_requests` | Concurrent requests to vLLM |

Default tasks: `math_500`, `gpqa:diamond`, `lcb:codegeneration_v6`, `aime25`

The launcher writes a `litellm_config.yaml` into the output directory and passes it to each `lighteval endpoint litellm` invocation.

### SWE-bench (`swebench/default.yaml`)

| Key | Description |
|---|---|
| `dataset_name` | HuggingFace dataset (default: `princeton-nlp/SWE-bench_Verified`) |
| `split` | Dataset split (default: `test`) |
| `max_workers` | Parallel Docker workers for evaluation |
| `run_id` | Run identifier for result grouping |
| `api_key` | API key passed to inference (default: `EMPTY`) |
| `prompt_style` | Prompt format for dataset creation: `style-3` (default), `style-2`, `full_file_gen`, `style-2-edits-only` |
| `predictions_path` | Path to an existing `.jsonl` to skip inference and go straight to evaluation |

The SWE-bench pipeline runs three steps automatically:

1. **`make_dataset`** — runs `swebench.inference.make_datasets.create_text_dataset` to build a text dataset with the configured prompt style. The raw HuggingFace dataset does not have the `text` column required by `run_api`.
2. **`inference`** — runs `swebench.inference.run_api` against the vLLM endpoint (via `OPENAI_BASE_URL`). Results are written as a `.jsonl` file and the step is resumable.
3. **`evaluation`** — runs `swebench.harness.run_evaluation` against the predictions file.

## Usage

### 1. Start vLLM server

```bash
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic \
  --host 0.0.0.0 \
  --port 8000 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-model-len 96000
```

### 2. Run benchmarks

```bash
llm-universal-eval
```

Hydra-style overrides are supported on the command line:

```bash
llm-universal-eval \
  model.name=RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic \
  model.host=127.0.0.1 \
  model.port=8001
```

### Selecting which benchmarks to run

The `benchmark` key accepts a list of harness names, individual task names, or `all` (default).
Task names are derived from the task string by replacing non-alphanumeric characters with `_`.

```bash
# Run everything (default)
llm-universal-eval benchmark=[all]

# Run only lm-eval harness
llm-universal-eval benchmark=[lm_eval]

# Run only lighteval harness
llm-universal-eval benchmark=[lighteval]

# Run only SWE-bench
llm-universal-eval benchmark=[swebench]

# Run specific tasks by derived name
llm-universal-eval benchmark=[ifeval,mmlu_pro]
llm-universal-eval benchmark=[math_500,gpqa_diamond,aime25]

# Mix harnesses and individual tasks
llm-universal-eval benchmark=[lm_eval,aime25]
```

### Customising the task list

The `tasks` list in each harness YAML defines the **default set** — the tasks that run when you select a harness by name (`benchmark=[lighteval]`) or run everything (`benchmark=[all]`).

It is not a gatekeeping list. If you explicitly name a task in `benchmark` that isn't in the default list, it runs anyway — routed automatically to the right harness **lighteval**, **lm-eval** or **swebench**.

```bash
# Runs gpqa:diamond even though it may not be in lighteval.tasks
llm-universal-eval 'benchmark=[gpqa:diamond]'

# Runs a custom lm-eval task directly
llm-universal-eval 'benchmark=[my_custom_lm_task]'
```

To permanently add a task to the default set, edit the YAML:

```yaml
# lighteval/default.yaml
tasks:
  - math_500
  - aime25
  - my_custom_task

# lm_eval/default.yaml
tasks:
  - ifeval
  - my_custom_lm_task
```

Or override the list entirely on the command line:

```bash
llm-universal-eval 'lighteval.tasks=[math_500,aime25]'
llm-universal-eval 'lm_eval.tasks=[ifeval]'
```

### Dry run

Print the commands that would be executed without running them:

```bash
llm-universal-eval dry_run=true
```

### Reusing a previous SWE-bench inference

```bash
llm-universal-eval \
  benchmark=[swebench] \
  swebench.predictions_path=/path/to/predictions.jsonl
```

## Output layout

```
results/
  config.yaml               # resolved Hydra config snapshot
  litellm_config.yaml       # generated LiteLLM config for lighteval
  lm_eval/<task_name>/      # lm-eval per-task results
  lighteval/<task_name>/    # lighteval per-task results
  swebench/                 # SWE-bench inference + evaluation outputs
```

## References

- [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
- [lighteval](https://github.com/neuralmagic/lighteval)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [vLLM](https://github.com/vllm-project/vllm)
