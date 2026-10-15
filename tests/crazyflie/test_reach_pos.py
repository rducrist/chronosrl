import jax
import jax.numpy as jnp
import pytest

from chronosrl.environments.crazyflow import get_config
from chronosrl.environments.crazyflow.drone_reach_pos_env import CrazyflowPositionEnv


def make_env(nr_envs=1):
    config = get_config("crazyflow_position")
    config.device = "cpu"
    return CrazyflowPositionEnv(config, 
                                nr_envs)


@pytest.mark.integration
def test_reset_at_fixed_position():
    env = make_env()
    point = jnp.array([-1.0, 1.0, 1.0])
    env.spawn_low = env.spawn_high = point
    env.goal_low = env.goal_high = point
    env.max_spawn_speed = 0.0

    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(42), 1))

    assert jnp.allclose(state.obs[:, 0:3], point)
    assert jnp.allclose(env.policy_observation(state.obs)[:, 0:3], 0.0)
    assert jnp.allclose(state.obs[:, 7:10], 0.0)
