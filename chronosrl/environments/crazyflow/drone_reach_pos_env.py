"""Goal-conditioned Crazyflie position-reaching environment on the crazyflow simulator."""

import crazyflow.sim.functional as F
import jax
import jax.numpy as jnp
from crazyflow.dynamics import load_params
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn, insert_fn_before, prepend_fn
from crazyflow.utils import leaf_replace
from flax import struct
from ml_collections import config_dict

from chronosrl.environments.crazyflow import domain_randomization as dr


@struct.dataclass
class State:
    """Batched state of all ``n`` parallel worlds."""

    data: SimData
    obs: jax.Array  # (n, 16) = [pos, quat, vel, ang_vel, goal]
    goal: jax.Array  # (n,3)
    reward: jax.Array  # (n, )
    key: jax.Array  # single PRNG key, split on every step for autoresets
    done: jax.Array  # (n, )
    info: dict  # seed (episode id), steps since reset, truncation
    metrics: dict


class CrazyflowPositionEnv:
    """Fly one Crazyflie per world to a random goal position with attitude commands.

    The first ``state_dim`` observation entries are the drone state ``[pos, quat, vel, ang_vel]``,
    the rest is the goal, which lives in the ``goal_indices`` slice of the state (the position).
    """

    state_dim, observation_size, action_size, goal_indices = 13, 16, 3, (0, 3)

    def __init__(self, config: config_dict.ConfigDict, nr_envs: int):
        """Build the batched simulator and precompute the action, bound and spawn ranges.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds, one drone each.
        """
        assert config.sim_freq % config.freq == 0, "sim_freq must be a multiple of freq"
        self.nr_envs = nr_envs
        self.episode_length = config.episode_length
        self.goal_radius = config.task.goal_radius
        self.rest_speed = config.task.rest_speed
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
            delay_steps = round(rand.action_delay * config.sim_freq)  # 0.04 s * 500 Hz = 20
            # Plugins must exist before build_default_data, so resets restore them
            self.sim.data = dr.add_delay_plugin(self.sim.data, delay_steps)
            self.sim.data = dr.add_wrench_plugin(self.sim.data)

            append_fn(self.sim.reset_pipeline, dr.build_randomize_mass_fn(rand.mass))
            append_fn(self.sim.reset_pipeline, dr.build_randomize_inertia_fn(rand.inertia))
            append_fn(
                self.sim.reset_pipeline, dr.build_randomize_rotor_dynamics_fn(rand.rotor_dynamics)
            )
            append_fn(
                self.sim.reset_pipeline, dr.build_randomize_thrust_curve_fn(rand.thrust_curve)
            )
            append_fn(
                self.sim.reset_pipeline, dr.build_randomize_torque_curve_fn(rand.torque_curve)
            )
            append_fn(
                self.sim.reset_pipeline, dr.build_randomize_wrench_fn(rand.force, rand.torque)
            )
            append_fn(self.sim.reset_pipeline, dr.build_randomize_delay_fn(delay_steps))
            prepend_fn(
                self.sim.step_pipeline,
                dr.build_delay_attitude_fn(self.n_substeps),
                name="delay",
            )
            insert_fn_before(
                self.sim.step_pipeline,
                "integration",
                dr.build_disturb_wrench_fn(rand.force_noise, rand.torque_noise),
                name="disturb_wrench",
            )
            self.sim.build_default_data()
            self.sim._reset = self.sim.build_reset_fn()
            self.sim._step = self.sim.build_step_fn()

        # Attitude commands: roll, pitch, yaw in radians, then collective thrust in newtons
        params = load_params(config.dynamics, config.drone)
        thrust_min, thrust_max = 4 * params["thrust_min"], 4 * params["thrust_max"]
        max_angle = config.action.max_angle
        self.action_low = jnp.array([-max_angle, -max_angle, thrust_min])
        self.action_high = jnp.array([max_angle, max_angle, thrust_max])

        # Episodes terminate when the drone leaves this box
        self.bounds_low = jnp.array(config.bounds.low)
        self.bounds_high = jnp.array(config.bounds.high)

        self.spawn_low = jnp.array(config.spawn.low)
        self.spawn_high = jnp.array(config.spawn.high)
        self.goal_low = jnp.array(config.goal.low)
        self.goal_high = jnp.array(config.goal.high)
        self.max_spawn_speed = config.spawn.max_speed

        # Spawn with rotors already at hover
        self.reset_rotor = config.reset_rotor
        hover_thrust = params["mass"] * jnp.linalg.norm(params["gravity_vec"]) / 4
        self.hover_rotor_vel = float(hover_thrust)
        # TODO there has to be a more elegant way
        # Solve k2 * rpm^2 + k1 * rpm + k0 = hover_thrust
        k0, k1, k2 = params["rpm2thrust"]
        hover_rpm = (-k1 + jnp.sqrt(k1**2 - 4 * k2 * (k0 - hover_thrust))) / (2 * k2)
        self.hover_rotor_vel = float(hover_rpm)

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Turn ``(..., 16)`` observations into ``(..., 13)`` policy inputs."""
        # Replace the absolute position of the drone with the relative position to the goal
        position = obs[..., :3]
        goal = obs[..., self.state_dim :]
        return jnp.concatenate([position - goal, obs[..., 3 : self.state_dim]], axis=-1)

    def observation(self, data: SimData, goal: jax.Array) -> jax.Array:
        """Build the ``(n, 16)`` observation ``[pos, quat, vel, ang_vel, goal]``."""
        states = data.states
        return jnp.concatenate(
            [states.pos[:, 0], states.quat[:, 0], states.vel[:, 0], states.ang_vel[:, 0], goal],
            axis=-1,
        ).astype(jnp.float32)

    def spawn(
        self, data: SimData, goal: jax.Array | None, key: jax.Array, mask: jax.Array | None = None
    ) -> tuple[SimData, jax.Array]:
        """Respawn the worlds in ``mask`` at a random position and velocity with a new goal.

        Positions, goals and velocity components are sampled uniformly from the configured boxes.

        Args:
            data: Simulator data of all worlds.
            goal: Current ``(n, 3)`` goals, kept where ``mask`` is False. Unused if no ``mask``.
            key: PRNG key.
            mask: ``(n,)`` boolean array of worlds to respawn, or None to respawn all of them.

        Returns:
            The updated simulator data and ``(n, 3)`` goals.
        """
        # Handles the randomization of goal and position
        pos_key, vel_key, goal_key = jax.random.split(key, 3)
        shape = (self.nr_envs, 1, 3)  # worlds, drones, xyz
        pos = jax.random.uniform(pos_key, shape, minval=self.spawn_low, maxval=self.spawn_high)
        vel = jax.random.uniform(
            vel_key, shape, minval=-self.max_spawn_speed, maxval=self.max_spawn_speed
        )
        new_goal = jax.random.uniform(
            goal_key, (self.nr_envs, 3), minval=self.goal_low, maxval=self.goal_high
        )

        data = self.sim._reset(data, self.sim.default_data, mask)
        spawn_state = {"pos": pos, "vel": vel}
        if self.reset_rotor:
            spawn_state["rotor_vel"] = jnp.full((self.nr_envs, 1, 4), self.hover_rotor_vel)
        data = data.replace(states=leaf_replace(data.states, mask, **spawn_state))

        goal = new_goal if mask is None else jnp.where(mask[:, None], new_goal, goal)

        return data, goal

    def reset(self, keys: jax.Array) -> State:
        """Spawn all worlds and return the initial state."""
        # Only keys[0] is used, spawn randomizes all worlds from a single key
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

    def step(self, state: State, action: jax.Array) -> State:
        """Advance one policy step and autoreset the worlds whose episode ended.

        Args:
            state: Current state.
            action: ``(n, 4)`` actions in ``[-1, 1]``, rescaled to attitude commands.

        Returns:
            The next state. For worlds that ended, ``obs`` and ``goal`` already belong to the new
            episode, while ``metrics`` are measured before the reset.
        """
        key, spawn_key = jax.random.split(state.key)

        # 1. Scale action and send to attitude control
        action = jnp.clip(action, -1.0, 1.0)
        scaled_action = self.action_low + 0.5 * (action + 1.0) * (
            self.action_high - self.action_low
        )

        # Drop yaw command
        yaw = jnp.zeros((self.nr_envs, 1))
        command = jnp.concatenate([scaled_action[:, :2], yaw, scaled_action[:, 2:]], axis=-1)
        data = F.attitude_control(state.data, command[:, None, :])

        # 2. Advance physics
        data = self.sim._step(data, self.n_substeps)

        # 3. Check how episode ended
        pos, vel = data.states.pos[:, 0], data.states.vel[:, 0]
        out_of_bounds = jnp.any((pos < self.bounds_low) | (pos > self.bounds_high), axis=-1)
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
            reward=jnp.zeros_like(state.reward),  # the trainer reads this as a cost; none here
            done=done.astype(jnp.float32),
            info={
                "seed": state.info["seed"] + done.astype(jnp.int32),
                "steps": jnp.where(done, 0.0, steps),
                "truncation": truncated.astype(jnp.float32),
            },
            metrics=metrics,
        )

    def training_metrics(self, state: State) -> dict[str, jax.Array]:
        """Return extra training metrics (none for this task)."""
        return {}

    def evaluation_step_metrics(
        self, pos: jax.Array, vel: jax.Array, goal: jax.Array
    ) -> dict[str, jax.Array]:
        """Per-world metrics for a single step.

        ``success``: within ``goal_radius`` of the goal. ``rest_success``: additionally slower than
        ``rest_speed``. ``goal_error``: distance to the goal.
        """
        distance = jnp.linalg.norm(goal - pos, axis=-1)
        speed = jnp.linalg.norm(vel, axis=-1)
        at_goal = distance < self.goal_radius
        at_rest = speed < self.rest_speed

        return {
            "success": at_goal.astype(jnp.float32),
            "rest_success": (at_goal & at_rest).astype(jnp.float32),
            "goal_error": distance,
        }
