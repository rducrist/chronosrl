import crazyflow.sim.functional as F
import jax
import jax.numpy as jnp
from crazyflow.dynamics import load_params
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace
from flax import struct


@struct.dataclass
class State:
    data: SimData
    obs: jax.Array  # (n, 16) = [pos, quat, vel, ang_vel, goal]
    goal: jax.Array  # (n,3)
    reward: jax.Array  # (n, )
    key: jax.Array  # single PRNG key, split on every step for autoresets
    done: jax.Array  # (n, )
    info: dict  # seed (episode id), steps since reset, truncation
    metrics: dict


class CrazyflowPositionEnv:
    state_dim, observation_size, action_size, goal_indices = 13, 16, 4, (0, 3)

    def __init__(self, config, nr_envs):
        self.nr_envs = nr_envs
        self.episode_length = config.episode_length
        self.goal_radius = config.goal_radius
        self.n_substeps = config.sim_freq // config.freq

        self.sim = Sim(
            n_worlds=nr_envs,
            n_drones=1,
            drone=config.drone,
            device=config.device,
            dynamics=config.dynamics,
            control="attitude",
            freq=config.sim_freq,
        )

        # Attitude commands: roll, pitch, yaw in radians, then collective thrust in newtons
        params = load_params(config.dynamics, config.drone)
        thrust_min, thrust_max = 4 * params["thrust_min"], 4 * params["thrust_max"]
        self.action_low = jnp.array([-jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, thrust_min])
        self.action_high = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, thrust_max])

        # Episodes terminate when the drone leaves this box
        self.bounds_low, self.bounds_high = (
            jnp.array([-4.0, -4.0, 0.0]),
            jnp.array([4.0, 4.0, 4.0]),
        )

        self.spawn_low, self.spawn_high = (
            jnp.array([-1.0, -1.0, 0.5]),
            jnp.array([1.0, 1.0, 1.5]),
        )
        self.goal_low, self.goal_high = (
            jnp.array([-1.0, -1.0, 0.5]),
            jnp.array([1.0, 1.0, 1.5]),
        )
        self.max_spawn_speed = 1.0

    def policy_observation(self, obs):
        # Replace the absolute position of the drone with the relative position to the goal
        position = obs[..., :3]
        goal = obs[..., self.state_dim :]
        return jnp.concatenate(
            [position - goal, obs[..., 3 : self.state_dim]], axis=-1
        )

    def observation(self, data: SimData, goal):
        states = data.states
        return jnp.concatenate(
            [
                states.pos[:, 0],
                states.quat[:, 0],
                states.vel[:, 0],
                states.ang_vel[:, 0],
                goal,
            ],
            axis=-1,
        ).astype(jnp.float32)

    def spawn(self, data, goal, key, mask=None):
        # Handles the randomization of goal and position
        pos_key, vel_key, goal_key = jax.random.split(key, 3)
        shape = (self.nr_envs, 1, 3)  # worlds, drones, xyz
        pos = jax.random.uniform(
            pos_key, shape, minval=self.spawn_low, maxval=self.spawn_high
        )
        vel = jax.random.uniform(
            vel_key, shape, minval=-self.max_spawn_speed, maxval=self.max_spawn_speed
        )
        new_goal = jax.random.uniform(
            goal_key, (self.nr_envs, 3), minval=self.goal_low, maxval=self.goal_high
        )

        data = self.sim._reset(data, self.sim.default_data, mask)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))

        goal = new_goal if mask is None else jnp.where(mask[:, None], new_goal, goal)

        return data, goal

    def reset(self, keys):
        key, spawn_key = jax.random.split(keys[0])
        data, goal = self.spawn(self.sim.default_data, None, spawn_key)
        zeros = jnp.zeros((self.nr_envs,), jnp.float32)
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        return State(
            data=data,
            obs=self.observation(data, goal),
            goal=goal,
            key=key,
            reward=zeros,
            done=zeros,
            info={
                "seed": jnp.zeros((self.nr_envs,), jnp.int32),
                "steps": zeros,
                "truncation": zeros,
            },
            metrics=self.evaluation_step_metrics(pos, vel, goal),
        )

    def step(self, state: State, action):
        key, spawn_key = jax.random.split(state.key)

        # 1. Scale action and send to attitude control
        action = jnp.clip(action, -1.0, 1.0)
        scaled_action = self.action_low + 0.5 * (action + 1.0) * (
            self.action_high - self.action_low
        )
        data = F.attitude_control(state.data, scaled_action[:, None, :])

        # 2. Advance physics
        data = self.sim._step(data, self.n_substeps)

        # 3. Check how episode ended
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        out_of_bounds = jnp.any(
            (pos < self.bounds_low) | (pos > self.bounds_high), axis=-1
        )
        not_finite = ~jnp.all(jnp.isfinite(pos) & jnp.isfinite(vel), axis=-1)
        terminated = out_of_bounds | not_finite
        steps = state.info["steps"] + 1.0
        truncated = (steps >= self.episode_length) & ~terminated
        done = terminated | truncated

        # Measured before the autoreset, against the goal the drone was chasing
        metrics = self.evaluation_step_metrics(pos, vel, state.goal)

        # 4. Autoreset the envs that ended
        data, goal = self.spawn(data, state.goal, spawn_key, mask=done)

        return state.replace(
            data=data,
            goal=goal,
            key=key,
            obs=self.observation(data, goal),
            reward=jnp.zeros_like(state.reward),
            done=done.astype(jnp.float32),
            info={
                "seed": state.info["seed"] + done.astype(jnp.int32),
                "steps": jnp.where(done, 0.0, steps),
                "truncation": truncated.astype(jnp.float32),
            },
            metrics=metrics,
        )

    def training_metrics(self, state):
        return {}

    def evaluation_step_metrics(self, pos, vel, goal):
        distance = jnp.linalg.norm(goal - pos, axis=-1)
        speed = jnp.linalg.norm(vel, axis=-1)
        at_goal = distance < self.goal_radius
        at_rest = speed < 0.05

        return {
            "success": at_goal.astype(jnp.float32),
            "rest_success": (at_goal & at_rest).astype(jnp.float32),
            "goal_error": distance,
        }
