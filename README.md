# Evaluation Benchmarks

Separate environment for running lm-evaluation-harness and lighteval benchmarks against the vLLM server.

## Setup

```bash
uv sync
uv pip install -e .
```

The package installs a console script named `llm-universal-eval`.

## Usage

### 1. Start vLLM server (in another terminal)

Start the vLLM in another terminal and possibly another environment (lighteval hardcoded dependencies are not quite compatible with current vLLM versions):

```bash
vllm serve salbeal/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-int4-AutoRound \
  --host 0.0.0.0 \
  --port 8000 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-model-len 96000
```

### 2. Run all benchmarks with the launcher script

Edit config/evaluation_config.yaml with the model name and API host/port, then run:

```bash
llm-universal-eval --config config/evaluation_config.yaml
```

The script launches benchmarks sequentially, one by one:

- `lm_eval`: GSM8K-Platinum, IFEval, MMLU-Pro
- `lighteval`: Math-500, GPQA Diamond, LiveCodeBench v6, AIME25
- `swebench`: SWE-bench inference, then SWE-bench evaluation

Generated outputs are written under `results/`. SWE-bench evaluation results are written to `results/swebench/evaluation_results/`. The script also materializes the LiteLLM config it passes to `lighteval` as `results/litellm_config.yaml`.

Useful options:

```bash
# Print commands without running them
llm-universal-eval --config config/evaluation_config.yaml --dry-run

# Run only lm-eval benchmarks
llm-universal-eval --config config/evaluation_config.yaml --only lm_eval

# Run only lighteval benchmarks
llm-universal-eval --config config/evaluation_config.yaml --only lighteval

# Run only SWE-bench
llm-universal-eval --config config/evaluation_config.yaml --only swebench

# Run individual benchmarks one by one
llm-universal-eval --config config/evaluation_config.yaml --benchmarks gsm8k_platinum
llm-universal-eval --config config/evaluation_config.yaml --benchmarks math_500 gpqa_diamond
llm-universal-eval --config config/evaluation_config.yaml --benchmarks swebench_inference
```

### 3. Run lm-eval benchmarks manually

```bash
cd /home/aidev/Workspace/evaluation
source .venv/bin/activate

# IfEval (0-shot)
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=salbeal/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-int4-AutoRound,base_url=http://localhost:8000/v1,num_concurrent=8,max_retries=3" \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=64000"

# MMLU-Pro (0-shot)
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=salbeal/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-int4-AutoRound,base_url=http://localhost:8000/v1,num_concurrent=8,max_retries=3" \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=64000"

# GSM8k Platinum (0-shot)
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=salbeal/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-int4-AutoRound,base_url=http://localhost:8000/v1,num_concurrent=8,max_retries=3" \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=64000"
```

### 4. Run lighteval reasoning benchmarks manually

```bash
# Create lighteval config (lighteval_model_config.yaml)
cat > lighteval_model_config.yaml << 'EOF'
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/salbeal/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-int4-AutoRound"
  base_url: "http://localhost:8000/v1"
  api_key: ""
  timeout: 2400
  concurrent_requests: 8
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 64000
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    presence_penalty: 1.5
    repetition_penalty: 1.0
    seed: 5678
EOF

# Run reasoning benchmarks
lighteval endpoint litellm lighteval_model_config.yaml "aime25|0,math_500|0,gpqa:diamond|0"
```

## References

- [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
- [lighteval](https://github.com/neuralmagic/lighteval)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [vLLM](https://github.com/vllm-project/vllm)
