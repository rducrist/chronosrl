"""Command-line entry point: assembles runner/algorithm/environment configs and trains."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from absl import app
from ml_collections import config_dict, config_flags

import wandb
from chronosrl.algorithms import ALGORITHMS
from chronosrl.config import load_toml
from chronosrl.environments import get_environment
from chronosrl.trainer import Trainer


def get_runner_config() -> config_dict.ConfigDict:
    """Return the default runner config (logging, wandb and checkpointing options)."""
    config = config_dict.ConfigDict()

    config.project_name = "chronosrl"
    config.exp_name = "default"
    config.run_name = f"{int(time.time())}"
    config.track_wandb = True
    config.wandb_entity = "rducristian-technical-university-of-munich"
    config.save_model = True

    return config


def load_experiment_config(
    experiment: dict[str, Any], algorithm_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split a loaded TOML experiment into (runner, algorithm, environment) override dicts."""
    # [algorithm] applies to every algorithm, [algorithm.<name>] only to that one
    algorithm = experiment.get("algorithm", {})
    algorithm_overrides = {key: value for key, value in algorithm.items() if key not in ALGORITHMS}
    algorithm_overrides.update(algorithm.get(algorithm_name, {}))

    return experiment.get("runner", {}), algorithm_overrides, experiment["environment"]


class Runner:
    """Builds the configs as absl config flags and launches training."""

    def __init__(self) -> None:
        """Load the experiment TOML, build the default configs it names, then define the flags."""
        # These are read from sys.argv before absl parses the flags, since they decide
        # which default configs the flags are built from.
        config_path = self.pop_argument("config", None)
        if config_path is None:
            raise ValueError("Pass the experiment file with --config=configs/<experiment>.toml")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        experiment = load_toml(config_path)

        # The CLI wins over the TOML, so one experiment file can run several algorithms
        algorithm_name = (
            self.pop_argument("algorithm.name", None) or experiment["algorithm"]["name"]
        )
        environment_name = experiment["environment"]["name"]
        get_algorithm_config, self.algorithm_class = ALGORITHMS[algorithm_name]
        self.environment = get_environment(environment_name)

        runner_config = get_runner_config()
        algorithm_config = get_algorithm_config(algorithm_name)
        runner_overrides, algorithm_overrides, environment_config = load_experiment_config(
            experiment, algorithm_name
        )
        runner_config.update(runner_overrides)
        algorithm_config.update(algorithm_overrides)
        # Keep the logged name equal to the algorithm that actually trains
        algorithm_config.name = algorithm_name
        environment_config = config_dict.ConfigDict(environment_config)

        self.runner_config_flag = config_flags.DEFINE_config_dict("runner", runner_config)
        self.algorithm_config_flag = config_flags.DEFINE_config_dict("algorithm", algorithm_config)
        self.environment_config_flag = config_flags.DEFINE_config_dict(
            "environment", environment_config
        )

    @staticmethod
    def pop_argument(name: str, default: str | None) -> str | None:
        """Remove `--name=value` from sys.argv and return its value, or `default` if absent."""
        for argument in sys.argv:
            if argument.startswith(f"--{name}="):
                sys.argv.remove(argument)
                return argument.split("=", 1)[1]
        return default

    def run(self):
        """Parse the flags with absl and run `train`."""
        app.run(self.train)

    def train(self, _):
        """Train with the parsed configs (absl entry point; argv is unused)."""
        config = config_dict.ConfigDict()
        config.runner = self.runner_config_flag.value
        config.algorithm = self.algorithm_config_flag.value
        config.environment = self.environment_config_flag.value
        print(config, flush=True)

        run_path = os.path.abspath(
            f"runs/{config.runner.project_name}/{config.runner.exp_name}/{config.runner.run_name}"
        )
        if config.runner.track_wandb:
            wandb.init(
                entity=config.runner.wandb_entity or None,
                project=config.runner.project_name,
                group=config.runner.exp_name,
                name=config.runner.run_name,
                config=config.to_dict(),
            )
            wandb.define_metric("*", step_metric="global_step")
        if config.runner.save_model:
            # Resolved config (TOML + CLI), read back by rollout.py
            os.makedirs(run_path, exist_ok=True)
            with open(os.path.join(run_path, "config.json"), "w") as file:
                json.dump(config.to_dict(), file, indent=2)

        train_env, eval_env = self.environment.create_env(config)
        Trainer(config, self.algorithm_class, train_env, eval_env, run_path).train()

        if config.runner.track_wandb:
            wandb.finish()
