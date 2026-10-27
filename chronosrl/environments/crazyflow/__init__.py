"""Crazyflow drone environments: config loading and env construction."""

from ml_collections import config_dict

from chronosrl.config import default_config_path, load_toml
from chronosrl.environments.crazyflow.drone_env import DroneEnv
from chronosrl.environments.crazyflow.drone_maze_env import DroneMazeEnv
from chronosrl.environments.crazyflow.wrappers import DeltaActions, EvalWrapper, Wrapper, ZeroYaw

# Maps [environment].name in the TOML to the env class
ENVS = {"crazyflow_position": DroneEnv, "crazyflow_maze": DroneMazeEnv}


def get_config(environment_name: str) -> config_dict.ConfigDict:
    """Return the environment config for ``environment_name``."""
    # Hyperparameters live in configs/<environment_name>.toml
    config = config_dict.ConfigDict(load_toml(default_config_path(environment_name))["environment"])
    config.name = environment_name
    return config


def make_env(config: config_dict.ConfigDict, nr_envs: int) -> DroneEnv | Wrapper:
    """Build the ``config.name`` env with ``nr_envs`` worlds, wrapped as ``config.action`` asks."""
    env = ENVS[config.name](config, nr_envs)
    if config.action.zero_yaw:
        env = ZeroYaw(env)

    if config.action.delta:
        env = DeltaActions(env, config)
    return env


def create_env(config: config_dict.ConfigDict) -> tuple[DroneEnv | Wrapper, EvalWrapper]:
    """Build the training env and the evaluation env from the full experiment config."""
    env_config = config.environment
    train_env = make_env(env_config, env_config.nr_envs)
    eval_env = EvalWrapper(make_env(env_config, env_config.nr_eval_envs))
    return train_env, eval_env
