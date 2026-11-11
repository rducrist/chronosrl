"""Goal-conditioned Crazyflie attitude-control environment on the crazyflow simulator."""

import crazyflow.sim.functional as F
import jax
import jax.numpy as jnp
from crazyflow.dynamics import load_params
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace
from flax import struct
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import config_dict

from chronosrl.environments.crazyflow import domain_randomization as dr


@struct.dataclass
class State:
    """Batched state of all ``n`` parallel worlds."""

    data: SimData
    obs: jax.Array  # (n, observation_size) = [drone state, goal]
    goal: jax.Array  # (n,3)
    task: dict  # per-episode constants of each world, e.g. a race track; empty for this task
    reward: jax.Array  # (n, )
    key: jax.Array  # single PRNG key, split on every step for autoresets
    done: jax.Array  # (n, )
    info: dict  # seed (episode id), steps since reset, truncation
    metrics: dict
    task: dict  # needed for the drone racing env


class DroneEnv:
    """Fly one Crazyflie per world to a random goal position with RPYT attitude commands.

    The first ``state_dim`` observation entries are the drone state
    ``[pos, rot_mat, vel, ang_vel]``, the rest is the goal, which lives in the
    ``goal_indices`` slice of the state (the position).
    """

    state_dim, observation_size, action_size, goal_indices = 18, 21, 4, (0, 3)

    def __init__(self, config: config_dict.ConfigDict, nr_envs: int):
        """Build the batched simulator and precompute the action, bound and spawn ranges.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds, one drone each.
        """
        assert config.sim_freq % config.freq == 0, "sim_freq must be a multiple of freq"
        self.nr_envs = nr_envs
        self.nr_waypoints = config.task.nr_waypoints
        self.observation_size = self.state_dim + self.nr_waypoints * 3


        self.episode_length = config.episode_length
        self.goal_radius = config.task.goal_radius
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

        # Randomization
        rand = config.randomization
        if rand.enabled:
            dr.add_randomization(self.sim, rand, self.n_substeps)

        self.sim.build_default_data()
        self.sim._reset = self.sim.build_reset_fn()
        self.sim._step = self.sim.build_step_fn()

        # Actions: roll, pitch, yaw in radians, then collective thrust in newtons
        params = load_params(config.dynamics, config.drone)
        thrust_min, thrust_max = 4 * params["thrust_min"], 4 * params["thrust_max"]
        max_angle = config.action.max_angle
        self.action_low = jnp.array([-max_angle, -max_angle, -max_angle, thrust_min])
        self.action_high = jnp.array([max_angle, max_angle, max_angle, thrust_max])

        # Episodes terminate when the drone leaves this box
        self.bounds_low = jnp.array(config.bounds.low)
        self.bounds_high = jnp.array(config.bounds.high)

        self.spawn_low = jnp.array(config.spawn.low)
        self.spawn_high = jnp.array(config.spawn.high)
        # The track env samples goals around nominal waypoints and has no goal box
        if "low" in config.goal:
            self.goal_low = jnp.array(config.goal.low)
            self.goal_high = jnp.array(config.goal.high)
        self.max_spawn_speed = config.spawn.max_speed

        # Spawn with rotors already at hover
        self.reset_rotor = config.reset_rotor
        hover_thrust = params["mass"] * jnp.linalg.norm(params["gravity_vec"]) / 4
        # TODO there has to be a more elegant way
        # Solve k2 * rpm^2 + k1 * rpm + k0 = hover_thrust
        k0, k1, k2 = params["rpm2thrust"]
        hover_rpm = (-k1 + jnp.sqrt(k1**2 - 4 * k2 * (k0 - hover_thrust))) / (2 * k2)
        self.hover_rotor_vel = float(hover_rpm)

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Turn ``(..., 21)`` observations into ``(..., 18)`` policy inputs."""
        # Replace the absolute position of the drone with the relative position to the goal
        pos = obs[..., :3]
        rot_mat = obs[..., 3:12]
        vel = obs[..., 12:15]
        ang_vel = obs[..., 15:18]  # is already in body frame
        goals = obs[..., self.state_dim :].reshape(obs.shape[:-1] + (self.nr_waypoints, 3))
        rel = (pos[..., None, :] - goals).reshape(obs.shape[:-1] + (3 * self.nr_waypoints,))

        return jnp.concatenate([rel, vel, rot_mat, ang_vel], axis=-1)

    def observation(self, data: SimData, goal: jax.Array) -> jax.Array:
        """Build the ``(n, 21)`` observation ``[pos, rot_mat, vel, ang_vel, goal]``."""
        states = data.states

        quat = states.quat
        # Rotation matrix instead of the quaternion: continuous and no q / -q ambiguity
        rotation = R.from_quat(quat)
        rot_mat = rotation.as_matrix().reshape(-1, 9)

        return jnp.concatenate(
            [
                states.pos[:, 0],
                rot_mat,
                states.vel[:, 0],
                states.ang_vel[:, 0],
                goal.reshape(goal.shape[0], -1),
            ],
            axis=-1,
        ).astype(jnp.float32)

    def sample_task(self, key: jax.Array) -> tuple[jax.Array, dict]:
        """Sample ``(n, K, 3)`` waypoint goals uniformly from the goal box, with an empty task."""
        goal = jax.random.uniform(
            key, (self.nr_envs, self.nr_waypoints, 3), minval=self.goal_low, maxval=self.goal_high
        )
        return goal, {}

    def spawn(
        self,
        data: SimData,
        goal: jax.Array | None,
        task: dict | None,
        key: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[SimData, jax.Array, dict]:
        """Respawn the worlds in ``mask`` at a random position and velocity with a new goal.

        Positions and velocity components are sampled uniformly from the configured boxes, goals
        and the task come from ``sample_task``.

        Args:
            data: Simulator data of all worlds.
            goal: Current ``(n, K, 3)`` goals, kept where ``mask`` is False. Unused if no ``mask``.
            task: Current task, kept where ``mask`` is False. Unused if no ``mask``.
            key: PRNG key.
            mask: ``(n,)`` boolean array of worlds to respawn, or None to respawn all of them.

        Returns:
            The updated simulator data, ``(n, K, 3)`` goals and task.
        """
        # Handles the randomization of goal and position
        pos_key, vel_key, goal_key = jax.random.split(key, 3)
        shape = (self.nr_envs, 1, 3)  # worlds, drones, xyz
        pos = jax.random.uniform(pos_key, shape, minval=self.spawn_low, maxval=self.spawn_high)
        vel = jax.random.uniform(
            vel_key, shape, minval=-self.max_spawn_speed, maxval=self.max_spawn_speed
        )

        new_goal, new_task = self.sample_task(goal_key)

        data = self.sim._reset(data, self.sim.default_data, mask)
        spawn_state = {"pos": pos, "vel": vel}
        if self.reset_rotor:
            spawn_state["rotor_vel"] = jnp.full((self.nr_envs, 1, 4), self.hover_rotor_vel)
        data = data.replace(states=leaf_replace(data.states, mask, **spawn_state))

        if mask is None:
            return data, new_goal, new_task

        def keep(new: jax.Array, old: jax.Array) -> jax.Array:
            # New value in the masked worlds, broadcast over the trailing axes
            return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old)

        return data, keep(new_goal, goal), jax.tree.map(keep, new_task, task)

    def reset(self, keys: jax.Array) -> State:
        """Spawn all worlds and return the initial state."""
        # Only keys[0] is used, spawn randomizes all worlds from a single key
        key, spawn_key = jax.random.split(keys[0])
        data, goal, task = self.spawn(self.sim.default_data, None, None, spawn_key)
        zeros = jnp.zeros((self.nr_envs,), jnp.float32)
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        return State(
            data=data,
            obs=self.observation(data, goal),
            goal=goal,
            task=task,
            key=key,
            reward=zeros,
            done=zeros,
            info={
                "seed": jnp.zeros((self.nr_envs,), jnp.int32),
                "steps": zeros,
                "truncation": zeros,
                "progress": jnp.zeros((self.nr_envs,), jnp.int32),
            },
            metrics={
                **self.evaluation_step_metrics(pos, vel, goal[:, 0], task),
                "sequence_done": zeros,
            },
        )

    def terminated(self, data: SimData, task: dict, goal: jax.Array) -> jax.Array:
        """Return the ``(n,)`` worlds whose drone left the bounds box or has a non-finite state.

        ``goal`` holds the ``(n, K, 3)`` goals chased in this step, unused here.
        """
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        out_of_bounds = jnp.any((pos < self.bounds_low) | (pos > self.bounds_high), axis=-1)
        not_finite = ~jnp.all(jnp.isfinite(pos) & jnp.isfinite(vel), axis=-1)
        return out_of_bounds | not_finite

    def apply_action(self, data: SimData, action: jax.Array) -> SimData:
        """Send ``(n, 4)`` RPYT actions in ``[-1, 1]`` as attitude commands and advance physics."""
        action = jnp.clip(action, -1.0, 1.0)
        command = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
        data = F.attitude_control(data, command[:, None, :])
        return self.sim._step(data, self.n_substeps)

    def next_goal(self, goal: jax.Array, task: dict, progress: jax.Array) -> jax.Array:
        """(n, 3) goal appended when the first goal is reached; repeats the last one."""
        return goal[:, -1]

    def sequence_length(self, task: dict) -> jax.Array:
        """(n,) goals to reach before the sequence is done."""
        return jnp.full(self.nr_envs, self.nr_waypoints)

    def step(self, state: State, action: jax.Array) -> State:
        """Advance one policy step and autoreset the worlds whose episode ended.

        Args:
            state: Current state.
            action: ``(n, 4)`` RPYT actions in ``[-1, 1]``, rescaled to attitude commands.

        Returns:
            The next state. For worlds that ended, ``obs`` and ``goal`` already belong to the new
            episode, while ``metrics`` are measured before the reset.
        """
        key, spawn_key = jax.random.split(state.key)

        # 1. Apply action and advance physics
        data = self.apply_action(state.data, action)

        # 2. Check how episode ended
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        terminated = self.terminated(data, state.task, state.goal)
        steps = state.info["steps"] + 1.0
        truncated = (steps >= self.episode_length) & ~terminated
        done = terminated | truncated

        progress = state.info["progress"]
        length = self.sequence_length(state.task)
        at_goal = jnp.linalg.norm(pos - state.goal[:, 0], axis=-1) < self.goal_radius
        passed = at_goal & ~terminated & (progress < length)
        appended = self.next_goal(state.goal, state.task, progress)[:, None]
        popped = jnp.concatenate([state.goal[:, 1:], appended], axis=1)
        goal = jnp.where(passed[:, None, None], popped, state.goal)
        progress = progress + passed

        # Measured before the autoreset, against the goal the drone was chasing
        metrics = {
            **self.evaluation_step_metrics(pos, vel, state.goal[:, 0], state.task),
            "sequence_done": (progress >= length).astype(jnp.float32),
        }

        # 3. Autoreset the envs that ended
        data, goal, task = self.spawn(data, goal, state.task, spawn_key, mask=done)

        return state.replace(
            data=data,
            goal=goal,
            task=task,
            key=key,
            obs=self.observation(data, goal),
            reward=jnp.zeros_like(state.reward),  # the trainer reads this as a cost; none here
            done=done.astype(jnp.float32),
            info={
                "seed": state.info["seed"] + done.astype(jnp.int32),
                "steps": jnp.where(done, 0.0, steps),
                "truncation": truncated.astype(jnp.float32),
                "progress": jnp.where(done, 0, progress),
            },
            metrics=metrics,
        )

    def training_metrics(self, state: State) -> dict[str, jax.Array]:
        """Return extra training metrics (none for this task)."""
        return {}

    def evaluation_step_metrics(
        self, pos: jax.Array, vel: jax.Array, goal: jax.Array, task: dict
    ) -> dict[str, jax.Array]:
        """Per-world metrics for a single step, against the current ``goal`` of the ``task``.

        ``success``: within ``goal_radius`` of the goal. ``goal_error``: distance to the goal.
        """
        distance = jnp.linalg.norm(goal - pos, axis=-1)
        at_goal = distance < self.goal_radius

        return {"success": at_goal.astype(jnp.float32), "goal_error": distance}
