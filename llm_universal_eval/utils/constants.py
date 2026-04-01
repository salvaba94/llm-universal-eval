LM_EVAL_BENCHMARKS = [
    {
        "task": "gsm8k_platinum_cot_llama",
        "timeout": 2400,
    },
    {
        "task": "ifeval",
        "timeout": 2400,
    },
    {
        "task": "mmlu_pro",
        "timeout": 3600,
    },
]

LIGHTEVAL_BENCHMARKS = [
    "math_500|0",
    "gpqa:diamond|0",
    "lcb:codegeneration_v6|0",
    "aime25|0",
]
