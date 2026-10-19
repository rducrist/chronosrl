import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.config import default_config_path, load_toml
from chronosrl.environments.crazyflow.drone_reach_pos_env import (
    CrazyflowPositionEnv,
    State,
)


class EvalWrapper:
    def __init__(self, env: CrazyflowPositionEnv):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, keys):
        state = self.env.reset(keys)
        return state.replace(
            info={
                **state.info,
                "episode_metrics": {
                    name: jnp.zeros_like(value) for name, value in state.metrics.items()
                },
                "active_episodes": jnp.ones_like(state.done),
                "episode_steps": jnp.zeros_like(state.done),
            }
        )

    def step(self, state: State, action):
        next_state = self.env.step(state, action)
        active = state.info["active_episodes"]
        return next_state.replace(
            info={
                **next_state.info,
                "episode_metrics": {
                    name: state.info["episode_metrics"][name] + value * active
                    for name, value in next_state.metrics.items()
                },
                "active_episodes": active * (1.0 - next_state.done),
                "episode_steps": state.info["episode_steps"] + active,
            }
        )

    def evaluation_metrics(self, state: State):
        episode = state.info["episode_metrics"]
        steps = state.info["episode_steps"]

        return {
            "time_at_goal": jnp.mean(episode["success"]),
            "goal_reached": jnp.mean(episode["success"] > 0),
            "time_at_goal_rest": jnp.mean(episode["rest_success"]),
            "goal_error": jnp.mean(episode["goal_error"] / jnp.maximum(steps, 1.0)),
            "episode_length": jnp.mean(steps),
        }


def get_config(environment_name):
    # Hyperparameters live in configs/<environment_name>.toml
    config = config_dict.ConfigDict(
        load_toml(default_config_path(environment_name))["environment"]
    )
    config.name = environment_name
    return config


def create_env(config):
    env_config = config.environment
    train_env = CrazyflowPositionEnv(env_config, env_config.nr_envs)
    eval_env = EvalWrapper(CrazyflowPositionEnv(env_config, env_config.nr_eval_envs))
    return train_env, eval_env