import types

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from chronosrl.relabeling import make_relabel_fn

# 1-D path along x: passes 0.5 fast at j = 1, comes back to 0.5 at rest at j = 3
X = [0.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
VX = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
J = len(X)


def relabel_many(rest_at_end: bool, nr_keys: int = 64) -> dict[str, np.ndarray]:
    """Relabel the path with only the future strategy (uniform in time), once per key."""
    observation = np.zeros((J, 21), np.float32)
    observation[:, 0], observation[:, 12] = X, VX
    transitions = {
        "observation": jnp.asarray(observation),
        "action": jnp.zeros((J, 4)),
        "discount": jnp.ones(J),
        "truncation": jnp.zeros(J),
        "seed": jnp.zeros(J),
        "cost": jnp.zeros(J),
    }
    env = types.SimpleNamespace(
        state_dim=18,
        goal_indices=(0, 3),
        velocity_indices=(12, 15),
        goal_radius=0.05,
        rest_speed=0.05,
        nr_waypoints=1,
        rest_at_end=rest_at_end,
    )
    config = config_dict.ConfigDict(
        {
            "goal_sequence_length": 1,
            "action_chunk_length": 1,
            "window": J,
            "goal_probabilities": (1.0, 0.0, 0.0),
            "future_goal_gamma": 1.0,
        }
    )
    relabel = jax.vmap(make_relabel_fn(config, env), in_axes=(None, 0))
    keys = jax.random.split(jax.random.PRNGKey(0), nr_keys)
    return jax.tree.map(np.asarray, relabel(transitions, keys))


def test_rest_goal_holds_the_velocity_and_every_future_goal_is_reached():
    batch = relabel_many(rest_at_end=True)

    assert batch["goal"].shape == (64, J, 6)  # position + velocity of the last waypoint
    assert batch["observation"].shape == (64, J, 21)  # the policy sees positions only
    assert np.all(batch["is_event"] == 1.0)

    # Anchor 0: the fast pass counts only for its own velocity, at rest only the return does
    goal_x, goal_vx, tau = batch["goal"][:, 0, 0], batch["goal"][:, 0, 3], batch["tau"][:, 0]
    fast = (goal_x == 0.5) & (goal_vx == 1.0)
    slow = (goal_x == 0.5) & (goal_vx == 0.0)
    assert fast.any() and slow.any()
    assert np.all(tau[fast] == 1)
    assert np.all(tau[slow] == 3)


def test_without_rest_the_goal_is_reached_at_the_first_touch():
    batch = relabel_many(rest_at_end=False)

    assert batch["goal"].shape == (64, J, 3)
    goal_x, tau = batch["goal"][:, 0, 0], batch["tau"][:, 0]
    assert (goal_x == 0.5).any()
    assert np.all(tau[goal_x == 0.5] == 1)
