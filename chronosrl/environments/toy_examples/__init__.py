"""Toy environments: env construction and evaluation wrapper."""

from typing import Any

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.environments.toy_examples.pendulum_env import PendulumEnv, State


class EvalWrapper:
    """Accumulates per-episode metrics on top of an env for evaluation.

    Only the first episode of each world is counted: ``active_episodes`` drops to 0 once it ends.
    """

    def __init__(self, env: PendulumEnv):
        """Wrap ``env``."""
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Forward any other attribute to the wrapped env."""
        return getattr(self.env, name)

    def reset(self, keys: jax.Array) -> State:
        """Reset the env and zero the episode accumulators in ``info``."""
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

    def step(self, state: State, action: jax.Array) -> State:
        """Step the env and add the step metrics to the episodes that are still active."""
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

    def evaluation_metrics(self, state: State) -> dict[str, jax.Array]:
        """Average the accumulated episode metrics over all evaluation worlds."""
        # Episode metrics are sums over steps, so time_at_goal is in policy steps
        episode = state.info["episode_metrics"]
        steps = state.info["episode_steps"]

        return {
            "time_at_goal": jnp.mean(episode["success"]),
            "goal_reached": jnp.mean(episode["success"] > 0),
            "goal_error": jnp.mean(episode["goal_error"] / jnp.maximum(steps, 1.0)),
            "episode_length": jnp.mean(steps),
        }


def create_env(config: config_dict.ConfigDict) -> tuple[PendulumEnv, EvalWrapper]:
    """Build the training env and the evaluation env from the full experiment config."""
    env_config = config.environment
    train_env = PendulumEnv(env_config, env_config.nr_envs)
    eval_env = EvalWrapper(PendulumEnv(env_config, env_config.nr_eval_envs))
    return train_env, eval_env
