"""FIFO replay buffer of flattened transitions, sampled as contiguous per-env windows."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class ReplayBuffer:
    """Stateless replay buffer; the data lives in a `buffer_state` dict passed in and out."""

    def __init__(
        self,
        dummy_transition: dict[str, Any],
        nr_envs: int,
        max_size: int,
        window: int,
        nr_windows: int,
    ) -> None:
        """Set up the buffer layout.

        Args:
            dummy_transition: Example single-env transition, used to fix the flattened layout.
            nr_envs: Number of parallel environments written per time step.
            max_size: Number of time steps kept per environment.
            window: Length of each sampled window of consecutive time steps.
            nr_windows: Number of windows per sample, each from a distinct environment.
        """
        flat, unflatten = ravel_pytree(dummy_transition)
        # Transitions are stored as flat vectors; vmap over the (time, env) axes.
        self.flatten = jax.vmap(jax.vmap(lambda x: ravel_pytree(x)[0]))
        self.unflatten = jax.vmap(jax.vmap(unflatten))
        self.shape = (max_size, nr_envs, flat.shape[0])
        self.nr_envs = nr_envs
        self.max_size = max_size
        self.window = window
        self.nr_windows = nr_windows

    def init(self) -> dict[str, jax.Array]:
        """Return an empty buffer state (`data` and the number of filled rows `position`)."""
        return {
            "data": jnp.zeros(self.shape, dtype=jnp.float32),
            "position": jnp.zeros((), dtype=jnp.int32),
        }

    def insert(
        self, buffer_state: dict[str, jax.Array], transitions: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """Append transitions with leading shape (time, nr_envs), dropping the oldest if full."""
        update = self.flatten(transitions)
        # Negative overflow: shift old rows out so the update fits at the end.
        roll = jnp.minimum(0, self.max_size - buffer_state["position"] - update.shape[0])
        data = jax.lax.cond(
            roll < 0,
            lambda: jnp.roll(buffer_state["data"], roll, axis=0),
            lambda: buffer_state["data"],
        )
        position = buffer_state["position"] + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        return {"data": data, "position": position + update.shape[0]}

    def sample(self, buffer_state: dict[str, jax.Array], key: jax.Array) -> dict[str, jax.Array]:
        """Sample windows of consecutive transitions, shaped (nr_windows, window, ...)."""
        key_envs, key_start = jax.random.split(key)
        envs = jax.random.choice(key_envs, self.nr_envs, shape=(self.nr_windows,), replace=False)
        # Starts in [0, position - window], so windows stay within the filled rows once
        # at least `window` rows have been written.
        starts = jax.random.randint(
            key_start,
            (self.nr_windows,),
            0,
            jnp.maximum(0, buffer_state["position"] - self.window) + 1,
        )
        indices = starts[:, None] + jnp.arange(self.window)[None, :]
        windows = jax.vmap(lambda data, index: data[index], in_axes=(1, 0))(
            buffer_state["data"][:, envs], indices
        )
        return self.unflatten(windows)
