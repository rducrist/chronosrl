from collections.abc import Callable

import jax
import pytest
from brax.envs.base import State
import jax.numpy as jnp

from ml_collections import config_dict

from chronosrl.config import default_config_path, load_toml
from chronosrl.environments.jaxgcrl.manipulation.arm_push_easy import ArmPushEasy

pytestmark = pytest.mark.integration

ResetFn = Callable[[jax.Array], State]  # read as [[input], output]
StepFn = Callable[[State, jax.Array], State]


@pytest.fixture(scope="module")
def env() -> ArmPushEasy:
    config = config_dict.ConfigDict(load_toml(default_config_path("arm_push_easy"))["environment"])
    return ArmPushEasy(config)


@pytest.fixture(scope="module")
def fns(env: ArmPushEasy) -> tuple[ResetFn, StepFn]:
    return jax.jit(env.reset), jax.jit(env.step)


# test the obs
def test_obs_layout(env: ArmPushEasy, fns: tuple[ResetFn, StepFn]) -> None:
    reset, step = fns
    state = reset(jax.random.PRNGKey(0))
    q = state.pipeline_state.q
    # are the obs in the correct shape
    assert state.obs.shape == (20,)
    # check for missmatch in state and obs
    assert jnp.allclose(q[:3], state.obs[:3])
    # check for same goal in obs and info
    assert jnp.allclose(state.obs[env.state_dim :], state.info["goal"])
    # check for the same goal in state and info
    state = step(state, jnp.zeros(env.action_size))
    assert jnp.allclose(state.pipeline_state.q[7:10], state.info["goal"])


# test the reset
def test_reset_distribution(env: ArmPushEasy) -> None:
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(1), 256))
    cube_xy, goal = states.obs[:, 0:2], states.info["goal"]

    # the cubes are spawned within the noise scale
    assert jnp.all(jnp.abs(cube_xy - env.sys.init_q[:2]) <= env.cube_noise_scale + 1e-6)
    # goals should be randomized
    assert jnp.std(goal[:, :2], axis=0).min() > 0.02


@pytest.mark.parametrize(
    "dist, expected",
    [(0.02, (1, 1, 1)), (0.05, (1, 1, 0)), (0.2, (0, 1, 0)), (0.5, (0, 0, 0))],)

# test if goal actually completed
def test_goal_completion(env, dist, expected):
    obs, goal = jnp.zeros(env.observation_size), jnp.array([dist, 0.0, 0.0])
    assert tuple(float(x) for x in env._compute_goal_completion(obs, goal)) == expected


# test if rollout is stable 
def test_random_rollout_is_stable(env):
    n = 10
    state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2), n))

    @jax.jit 
    def rollout(state, key):
        def body(state, k):
            # sample random action, update the new state 
            action = jax.random.uniform(k, (n, env.action_size), minval=-1, maxval=1)
            state = jax.vmap(env.step)(state, action)
            return state, state.obs
        
        return jax.lax.scan(body, state, jax.random.split(key, env.episode_length))
    
    _, obs = rollout(state, jax.random.PRNGKey(3))
    assert jnp.all(jnp.isinf(obs))
    assert jnp.all(obs[:, 2] > 0.0)

