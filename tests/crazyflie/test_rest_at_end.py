import jax
import jax.numpy as jnp
import pytest

from chronosrl.environments.crazyflow import get_config
from chronosrl.environments.crazyflow.drone_env import DroneEnv


def sequence_done_after_one_step(speed: float, rest_at_end: bool) -> float:
    """Spawn on the goal, freeze the drone at ``speed`` along x and take one step."""
    config = get_config("crazyflow_position")
    config.device = "cpu"
    config.task.nr_waypoints = 1
    config.task.rest_at_end = rest_at_end
    env = DroneEnv(config, 1)
    point = jnp.array([0.0, 0.0, 1.0])
    env.spawn_low = env.spawn_high = env.goal_low = env.goal_high = point
    env.max_spawn_speed = 0.0
    env.apply_action = lambda data, action: data  # freeze the drone

    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    vel = jnp.zeros_like(state.data.states.vel).at[..., 0].set(speed)
    state = state.replace(data=state.data.replace(states=state.data.states.replace(vel=vel)))
    state = env.step(state, jnp.zeros((1, 4)))
    return float(state.metrics["sequence_done"][0])


@pytest.mark.integration
def test_last_waypoint_needs_rest():
    assert sequence_done_after_one_step(speed=0.0, rest_at_end=True) == 1.0
    assert sequence_done_after_one_step(speed=1.0, rest_at_end=True) == 0.0
    assert sequence_done_after_one_step(speed=1.0, rest_at_end=False) == 1.0
