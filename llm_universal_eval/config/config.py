from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError


class ModelConfig(BaseModel):
    name: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_gen_toks: int = 64000
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0


class LMEvalConfig(BaseModel):
    tasks: list[str] = []
    max_length: int = 96000
    num_concurrent: int = 128
    max_retries: int = 3
    tokenized_requests: bool = False
    tokenizer_backend: str | None = None


class LightEvalConfig(BaseModel):
    tasks: list[str] = []
    provider: str = "hosted_vllm"
    api_key: str = ""
    model_name: str | None = None
    timeout: int = 2400
    concurrent_requests: int = 128


class SWEBenchConfig(BaseModel):
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    max_workers: int = 4
    run_id: str = "eval"
    api_key: str = "EMPTY"
    predictions_path: str | None = None
    prompt_style: str = "style-3"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_dir: str = "results"
    seed: int = 42
    benchmark: list[str] = ["all"]
    dry_run: bool = False
    model: ModelConfig = ModelConfig()
    lm_eval: LMEvalConfig = LMEvalConfig()
    lighteval: LightEvalConfig = LightEvalConfig()
    swebench: SWEBenchConfig = SWEBenchConfig()

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> AppConfig:
        OmegaConf.resolve(cfg)
        data = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(data, dict):
            raise ValueError("Config file is not a mapping object.")
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            raise ValueError(f"Config validation failed: {error}") from error