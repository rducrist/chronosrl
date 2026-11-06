import types

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from chronosrl.relabeling import make_relabel_fn

# 1-D path along x: passes 0.5 at j = 1 and comes back to it at j = 3
X = [0.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
J = len(X)

# Touches 1 at j = 2 without staying, then holds it from j = 4 on
X_HOLD = [0.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def relabel_many(
    x: list[float] = X, nr_waypoints: int = 1, m: int = 1, nr_keys: int = 64
) -> dict[str, np.ndarray]:
    """Relabel the path with only the future strategy (uniform in time), once per key."""
    J = len(x)
    observation = np.zeros((J, 18 + 3 * nr_waypoints), np.float32)
    observation[:, 0] = x
    transitions = {
        "observation": jnp.asarray(observation),
        "action": jnp.zeros((J, 4)),
        "discount": jnp.ones(J),
        "truncation": jnp.zeros(J),
        "seed": jnp.zeros(J),
        "cost": jnp.zeros(J),
    }
    env = types.SimpleNamespace(
        state_dim=18, goal_indices=(0, 3), goal_radius=0.05, nr_waypoints=nr_waypoints
    )
    config = config_dict.ConfigDict(
        {
            "goal_sequence_length": m,
            "action_chunk_length": 1,
            "window": J,
            "goal_probabilities": (1.0, 0.0, 0.0),
            "future_goal_gamma": 1.0,
        }
    )
    relabel = jax.vmap(make_relabel_fn(config, env), in_axes=(None, 0))
    keys = jax.random.split(jax.random.PRNGKey(0), nr_keys)
    return jax.tree.map(np.asarray, relabel(transitions, keys))


def test_goal_is_reached_at_the_first_touch():
    batch = relabel_many()

    assert batch["goal"].shape == (64, J, 3)
    goal_x, tau = batch["goal"][:, 0, 0], batch["tau"][:, 0]
    assert (goal_x == 0.5).any()
    assert np.all(tau[goal_x == 0.5] == 1)


def test_last_waypoint_is_reached_when_its_tail_is_held():
    nr_waypoints, m = 2, 3
    batch = relabel_many(X_HOLD, nr_waypoints, m)
    T = len(X_HOLD) - (m - 1)

    # Critic goal [g1 | g2, s2, s3], the policy sees [g1, g2]
    assert batch["goal"].shape == (64, T, 3 * (nr_waypoints - 1 + m))
    assert batch["observation"].shape == (64, T, 18 + 3 * nr_waypoints)
    # Future goals were achieved later in the episode, so every one is reached
    assert np.all(batch["is_event"] == 1.0)

    # x of the tail, from anchor 0
    tail_x, tau = batch["goal"][:, 0, 3::3], batch["tau"][:, 0]
    # Holding 1 starts at j = 4, the touch at j = 2 does not count
    held = np.all(tail_x == [1.0, 1.0, 1.0], axis=-1)
    assert held.any()
    assert np.all(tau[held] == 4)
    # The pass through 0.5 matches only where it happened
    passing = np.all(tail_x == [0.5, 1.0, 0.5], axis=-1)
    assert passing.any()
    assert np.all(tau[passing] == 1)
