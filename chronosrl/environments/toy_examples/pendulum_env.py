"""Goal-conditioned pendulum: swing to a goal angle with bounded torque."""

import jax
import jax.numpy as jnp
from flax import struct
from ml_collections import ConfigDict


@struct.dataclass
class State:
    """Batched state of all ``n`` parallel worlds."""

    theta: jax.Array  # (n,) angle, 0 = upright, wrapped to [-pi, pi)
    theta_dot: jax.Array  # (n,) angular velocity
    goal: jax.Array  # (n,) goal angle
    obs: jax.Array  # (n, 5) = [cos(theta), sin(theta), theta_dot, cos(goal), sin(goal)]
    reward: jax.Array  # (n,)
    key: jax.Array  # single PRNG key, split on every step for autoresets
    done: jax.Array  # (n,)
    info: dict  # seed (episode id), steps since reset, truncation
    metrics: dict


class PendulumEnv:
    """Swing a torque-limited pendulum to a goal angle, with ``theta = 0`` upright.

    The first ``state_dim`` observation entries are ``[cos(theta), sin(theta), theta_dot]``, the
    rest is the goal ``[cos(goal), sin(goal)]``, which lives in the ``goal_indices`` slice.
    """

    state_dim, observation_size, action_size, goal_indices = 3, 5, 1, (0, 2)

    def __init__(self, config: ConfigDict, nr_envs: int) -> None:
        """Read the physics, spawn and goal ranges.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds.
        """
        self.nr_envs = nr_envs
        self.episode_length = config.episode_length
        self.goal_radius = config.task.goal_radius  # Read by relabeling fn
        self._gravity = config.gravity
        self._max_torque = config.max_torque
        self._max_speed = config.max_speed
        self._mass = config.mass
        self._pendulum_length = config.pendulum_length
        self._dt = config.dt
        # spawn.max_speed (for theta_dot), goal.low / goal.high (angle range in radians)
        self._spawn_max_speed = config.spawn.max_speed
        self._goal_high = config.goal.high
        self._goal_low = config.goal.low

    def wrap_angle(self, theta: jax.Array) -> jax.Array:
        """Wrap angles to ``[-pi, pi)``."""
        return ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Return the observation unchanged as the policy input."""
        return obs

    def observation(self, theta: jax.Array, theta_dot: jax.Array, goal: jax.Array) -> jax.Array:
        """Build the ``(n, 5)`` observation from ``(n,)`` angles, velocities and goal angles."""
        return jnp.stack(
            [jnp.cos(theta), jnp.sin(theta), theta_dot, jnp.cos(goal), jnp.sin(goal)], axis=-1
        ).astype(jnp.float32)

    def dynamics(
        self, theta: jax.Array, theta_dot: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Advance one step with semi-implicit Euler; ``action`` is ``(n, 1)`` in ``[-1, 1]``."""
        u = self._max_torque * jnp.clip(action[..., 0], -1, 1)
        theta_dot_new = (
            theta_dot
            + (
                3 * self._gravity / (2 * self._pendulum_length) * jnp.sin(theta)
                + 3 / (self._mass * self._pendulum_length**2) * u
            )
            * self._dt
        )
        theta_dot_new = jnp.clip(theta_dot_new, -self._max_speed, self._max_speed)
        theta_new = self.wrap_angle(theta + theta_dot_new * self._dt)

        return theta_new, theta_dot_new

    def step(self, state: State, action: jax.Array) -> State:
        """Advance one policy step and autoreset the worlds whose episode ended.

        Args:
            state: Current state.
            action: ``(n, 1)`` torques in ``[-1, 1]``, scaled by ``max_torque``.

        Returns:
            The next state. For worlds that ended, ``obs`` and ``goal`` already belong to the new
            episode, while ``metrics`` are measured before the reset.
        """
        key, spawn_key = jax.random.split(state.key)

        theta, theta_dot = self.dynamics(state.theta, state.theta_dot, action)

        steps = state.info["steps"] + 1
        truncated = steps >= self.episode_length
        done = truncated

        metrics = self.step_metrics(theta, state.goal)
        theta, theta_dot, goal = self.spawn(theta, theta_dot, state.goal, spawn_key, mask=done)

        return state.replace(
            theta=theta,
            theta_dot=theta_dot,
            goal=goal,
            obs=self.observation(theta, theta_dot, goal),
            reward=jnp.zeros_like(state.reward),
            key=key,
            done=done.astype(jnp.float32),
            info={
                "seed": state.info["seed"] + done.astype(jnp.int32),
                "steps": jnp.where(done, 0.0, steps),
                "truncation": truncated.astype(jnp.float32),
            },
            metrics=metrics,
        )

    def spawn(
        self,
        theta: jax.Array | None,
        theta_dot: jax.Array | None,
        goal: jax.Array | None,
        key: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Respawn the worlds in ``mask`` with a random angle, velocity and goal.

        Args:
            theta: Current ``(n,)`` angles, kept where ``mask`` is False. Unused if no ``mask``.
            theta_dot: Current ``(n,)`` angular velocities, as ``theta``.
            goal: Current ``(n,)`` goal angles, as ``theta``.
            key: PRNG key.
            mask: ``(n,)`` boolean array of worlds to respawn, or None to respawn all of them.

        Returns:
            The ``(n,)`` angles, angular velocities and goal angles.
        """
        theta_key, theta_dot_key, goal_key = jax.random.split(key, 3)

        new_theta = jax.random.uniform(
            theta_key, shape=(self.nr_envs,), minval=-jnp.pi, maxval=jnp.pi
        )
        new_theta_dot = jax.random.uniform(
            theta_dot_key,
            shape=(self.nr_envs,),
            minval=-self._spawn_max_speed,
            maxval=self._spawn_max_speed,
        )
        new_goal = jax.random.uniform(
            goal_key, shape=(self.nr_envs,), minval=self._goal_low, maxval=self._goal_high
        )

        if mask is None:
            return new_theta, new_theta_dot, new_goal

        theta = jnp.where(mask, new_theta, theta)
        theta_dot = jnp.where(mask, new_theta_dot, theta_dot)
        goal = jnp.where(mask, new_goal, goal)
        return theta, theta_dot, goal

    def training_metrics(self, state: State) -> dict[str, jax.Array]:
        """Return extra training metrics (none for this task)."""
        return {}

    def step_metrics(self, theta: jax.Array, goal: jax.Array) -> dict[str, jax.Array]:
        """Per-world metrics for a single step.

        ``success``: within ``goal_radius`` of the goal, measured like the relabeling as the
        distance between ``(cos, sin)`` points. ``goal_error``: absolute angle to the goal.
        """
        achieved = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        target = jnp.stack([jnp.cos(goal), jnp.sin(goal)], axis=-1)
        distance = jnp.linalg.norm(achieved - target, axis=-1)

        return {
            "success": (distance <= self.goal_radius).astype(jnp.float32),
            "goal_error": jnp.abs(self.wrap_angle(theta - goal)),
        }

    def reset(self, keys: jax.Array) -> State:
        """Spawn all worlds and return the initial state."""
        # Only keys[0] is used, spawn randomizes all worlds from a single key
        key, spawn_key = jax.random.split(keys[0])
        theta, theta_dot, goal = self.spawn(None, None, None, spawn_key)
        zeros = jnp.zeros((self.nr_envs,), jnp.float32)
        return State(
            theta=theta,
            theta_dot=theta_dot,
            goal=goal,
            obs=self.observation(theta, theta_dot, goal),
            reward=zeros,
            key=key,
            done=zeros,
            info={
                "seed": jnp.zeros((self.nr_envs,), jnp.int32),
                "steps": zeros,
                "truncation": zeros,
            },
            metrics=self.step_metrics(theta, goal),
        )
