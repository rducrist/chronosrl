import jax
from crazyflow.sim.data import SimData
from flax import struct


@struct.dataclass
class State:
    data : SimData
    obs : jax.Array     # (n, 16) = [pos, quat, vel, ang_vel, goal]
    goal:  jax.Array    # (n,3) 
    reward: jax.Array   # (n, )
    done: jax.Array     # (n, )
    info: dict          # (seed, steps, truncations, what else?)

class CrazyflowPositionEnv:
    state_dim, observation_size, action_size, goal_indices = 13, 16, 4, (0,3)

    def __init__(self, config, nr_envs, evaluate = False):
        pass

    def policy_observation(self, obs):
        pass

    def reset(self, key):
        pass

    def step(self, state, action):
        pass

    def training_metrics(self, state):
        pass

    def evaluation_metric(self, state):
        pass

def create_env(config):
    env = config.environment
    return (CrazyflowPositionEnv(env, env.nr_envs),
            CrazyflowPositionEnv(env, env.nr_eval_envs, evaluate=True))
