import jax
import jax.numpy as jnp
import pytest

from chronosrl.environments.crazyflow import get_config
from chronosrl.environments.crazyflow.drone_env import DroneEnv


def make_env(nr_envs: int = 1) -> DroneEnv:
    # Unwrapped, so the attribute overrides in the tests reach the env
    config = get_config("crazyflow_position")
    config.device = "cpu"
    return DroneEnv(config, nr_envs)


@pytest.mark.integration
def test_reset_at_fixed_position():
    env = make_env()
    point = jnp.array([-1.0, 1.0, 1.0])
    env.spawn_low = env.spawn_high = point
    env.goal_low = env.goal_high = point
    env.max_spawn_speed = 0.0

    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(42), 1))

    # obs = [pos 0:3, rot_mat 3:12, vel 12:15, ang_vel 15:18, goal 18:21]
    assert jnp.allclose(state.obs[:, 0:3], point)
    assert jnp.allclose(env.policy_observation(state.obs)[:, 0:3], 0.0)
    assert jnp.allclose(state.obs[:, 12:15], 0.0)
