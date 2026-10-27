"""Env wrappers that change actions or observations on top of a crazyflow env."""

from typing import Any

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from chronosrl.environments.crazyflow.drone_env import DroneEnv, State


class Wrapper:
    """Base wrapper: every attribute it does not define is forwarded to the wrapped env."""

    def __init__(self, env: DroneEnv):
        """Wrap ``env``."""
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Forward any other attribute to the wrapped env."""
        return getattr(self.env, name)


# region ZeroYaw
class ZeroYaw(Wrapper):
    """Hold the yaw command at 0, so the policy outputs ``[roll, pitch, thrust]``."""

    def __init__(self, env: DroneEnv):
        """Wrap ``env`` and drop yaw from its action."""
        super().__init__(env)
        self.action_size = env.action_size - 1

    def step(self, state: State, action: jax.Array) -> State:
        """Insert a zero yaw at index 2 and step the wrapped env."""
        return self.env.step(state, jnp.insert(action, 2, 0.0, axis=-1))


# region DeltaAction
class DeltaActions(Wrapper):
    """Let the policy output changes of up to ``max_delta`` to the last applied action.

    The last action is added to the drone state, so the observation becomes
    ``[drone state, last_action, goal]``: ``state_dim`` and ``observation_size`` grow by
    ``action_size`` while ``goal_indices`` stay the same.
    """

    def __init__(self, env: DroneEnv, config: ConfigDict):
        """Wrap ``env`` with delta actions of at most ``config.action.max_delta`` per step."""
        super().__init__(env)

        self.max_delta = config.action.max_delta
        self.state_dim = env.state_dim + env.action_size
        self.observation_size = env.observation_size + env.action_size

    def add_last_action(self, obs: jax.Array, last_action: jax.Array) -> jax.Array:
        """Insert ``last_action`` between the drone state and the goal of the wrapped ``obs``."""
        n = self.env.state_dim
        return jnp.concatenate([obs[..., :n], last_action, obs[..., n:]], axis=-1)

    def reset(self, keys: jax.Array) -> State:
        """Reset the wrapped env with a zero last action."""
        state = self.env.reset(keys)
        last_action = jnp.zeros((self.env.nr_envs, self.env.action_size))
        return state.replace(
            obs=self.add_last_action(state.obs, last_action),
            info={**state.info, "last_action": last_action},
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Add the action to the last one, step the wrapped env and store the result."""
        action = state.info["last_action"] + self.max_delta * jnp.clip(action, -1.0, 1.0)
        action = jnp.clip(action, -1.0, 1.0)
        next_state = self.env.step(state, action)
        last_action = jnp.where(next_state.done[:, None] > 0, 0.0, action)
        return next_state.replace(
            obs=self.add_last_action(next_state.obs, last_action),
            info={**next_state.info, "last_action": last_action},
        )

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Wrapped policy inputs with the last action appended."""
        n, m = self.env.state_dim, self.env.action_size
        inner_obs = jnp.concatenate([obs[..., :n], obs[..., n + m :]], axis=-1)
        return jnp.concatenate(
            [self.env.policy_observation(inner_obs), obs[..., n : n + m]], axis=-1
        )


# region Eval
class EvalWrapper(Wrapper):
    """Accumulates per-episode metrics on top of an env for evaluation.

    Only the first episode of each world is counted: ``active_episodes`` drops to 0 once it ends.
    """

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
        # Episode metrics are sums over steps, so the time_at_goal values are in policy steps
        episode = state.info["episode_metrics"]
        steps = state.info["episode_steps"]

        metrics = {
            "time_at_goal": jnp.mean(episode["success"]),
            "goal_reached": jnp.mean(episode["success"] > 0),
            "time_at_goal_rest": jnp.mean(episode["rest_success"]),
            "goal_error": jnp.mean(episode["goal_error"] / jnp.maximum(steps, 1.0)),
            "episode_length": jnp.mean(steps),
        }
        # Only the maze env reports wall crashes
        if "crashed" in episode:
            metrics["crash_rate"] = jnp.mean(episode["crashed"] > 0)
        return metrics
