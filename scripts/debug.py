"""Debug script for llm-universal-eval main evaluation run."""

import os

from hydra import compose, initialize
from omegaconf import OmegaConf


# Override configuration options
overrides = [
    # Model settings:
    # "model.name=salbeal/...",
    "model.host=localhost",
    "model.port=8000",

    "output_dir=results",
    "seed=42",
    "benchmark=['swebench']",
    "dry_run=false",
]


if __name__ == "__main__":
    with initialize(config_path="pkg://llm_universal_eval.config", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

        print("-" * 60)
        print("Composed configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("-" * 60)

        # Import the main entry point from llm_universal_eval.
        # The @hydra.main decorator wraps the original function.
        # We extract the underlying function with __wrapped__ to bypass Hydra's CLI parsing.
        from llm_universal_eval.cli import main as hydra_main

        real_main = hydra_main.__wrapped__

        # Now call the underlying main function directly with our configuration.
        real_main(cfg)
