LM_EVAL_BENCHMARKS = [
    {
        "name": "gsm8k_platinum",
        "task": "gsm8k_platinum_cot_llama",
        "num_fewshot": 0,
        "timeout": 2400,
    },
    {
        "name": "ifeval",
        "task": "ifeval",
        "num_fewshot": None,
        "timeout": 2400,
    },
    {
        "name": "mmlu_pro",
        "task": "mmlu_pro_chat",
        "num_fewshot": 0,
        "timeout": 3600,
    },
]

LIGHTEVAL_BENCHMARKS = [
    {
        "name": "math_500",
        "task": "math_500|0",
    },
    {
        "name": "gpqa_diamond",
        "task": "gpqa:diamond|0",
    },
    {
        "name": "livecodebench_v6",
        "task": "lcb:codegeneration_v6|0",
    },
    {
        "name": "aime25",
        "task": "aime25|0",
    },
]

SWEBENCH_BENCHMARKS = ["swebench_inference", "swebench_evaluation"]

BENCHMARK_CHOICES = [
    *(benchmark["name"] for benchmark in LM_EVAL_BENCHMARKS),
    *(benchmark["name"] for benchmark in LIGHTEVAL_BENCHMARKS),
    *SWEBENCH_BENCHMARKS,
]

HARNESS_CHOICES = ["lm_eval", "lighteval", "swebench", "all"]