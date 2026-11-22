import dataclasses
import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx import base as brax_mjx
import mujoco
from mujoco import mjx
import jax
from jax import numpy as jnp

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "assets"
)

# MuJoCo Warp buffer sizes per env; overflowing contacts are dropped with a printed warning
_NCONMAX_PER_ENV = 64
_NJMAX_PER_ENV = 256


def asset_path(name: str) -> str:
    """Absolute path to an MJCF file under srl/envs/assets/."""
    return os.path.join(_ASSETS_DIR, name)


class ArmEnvs(PipelineEnv):
    def __init__(self, config, backend="mjx", **kwargs):
        """``config`` is the ``environment`` section of the experiment config."""
        self.config = config
        self.nr_waypoints = 1
        # Configure environment information (e.g. env name, noise scale, observation dimension, goal indices) and load XML
        self._set_environment_attributes()
        xml_path = self._get_xml_path()
        mj_model = mjcf.load_mjmodel(xml_path)

        # Configure backend
        mj_model.opt.timestep = config.timestep
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 12
        # Physics runs on MuJoCo Warp; the brax sys only carries model metadata and the viewer
        sys = mjcf.load_model(mj_model)
        self._mj_model = mj_model
        self._warp_model = mjx.put_model(mj_model, impl="warp")
        # Under vmap all envs share one contact buffer, so size it for the largest batch
        self._naconmax = _NCONMAX_PER_ENV * max(config.nr_envs, config.nr_eval_envs)
        self.n_frames = config.n_frames
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)

        # Initialize brax PipelineEnv
        if backend != "mjx":
            raise Exception("Use the mjx backend for stability/reasonable speed.")
        super().__init__(sys=sys, backend=backend, **kwargs)

    def pipeline_init(self, q: jax.Array, qd: jax.Array) -> base.State:
        """brax's mjx pipeline.init, on the Warp model."""
        data = mjx.make_data(
            self._mj_model, impl="warp", naconmax=self._naconmax, njmax=_NJMAX_PER_ENV
        )
        data = mjx.forward(self._warp_model, data.replace(qpos=q, qvel=qd))
        return self._brax_state(data)

    def pipeline_step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """n_frames Warp physics steps holding ``action`` as the control."""
        data = mjx.Data(**{f.name: getattr(pipeline_state, f.name) for f in dataclasses.fields(mjx.Data)})
        data = data.replace(ctrl=action)
        data = jax.lax.fori_loop(
            0, self._n_frames, lambda _, d: mjx.step(self._warp_model, d), data
        )
        return self._brax_state(data)

    def _brax_state(self, data: mjx.Data) -> base.State:
        """Adds brax's q, qd and per-link x, xd views to ``data``, as brax's mjx pipeline does."""
        x = base.Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = base.Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:] - data.subtree_com[self._mj_model.body_rootid[1:]]
        xd = base.Transform.create(pos=offset).vmap().do(cvel)
        # Warp keeps its contacts in data._impl, and nothing here reads brax's contact view
        return brax_mjx.State(q=data.qpos, qd=data.qvel, x=x, xd=xd, contact=None, **data.__dict__)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        # Initialize simulator state
        rng, subkey = jax.random.split(rng)
        q, qd = self._get_initial_state(
            subkey
        )  # Injects noise to avoid overfitting/open loop control
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0

        # Sample a goal and fill info variable
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        goal = self._get_initial_goal(pipeline_state, subkey1)
        pipeline_state = self._update_goal_visualization(pipeline_state, goal)
        info = {
            "seed": 0,  # Seed is required, but fill it with a dummy value
            "goal": goal,
            "goal_progress": jnp.array(0, dtype=jnp.int32),
            "timestep": 0.0,
            "postexplore_timestep": jax.random.uniform(
                subkey2
            ),  # Assumes timestep is normalized between 0 and 1
        }

        # Get components for state (observation, reward, metrics)
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Run mujoco step
        pipeline_state0 = state.pipeline_state
        if "EEF" in self.env_name:
            action = self._convert_action_to_actuator_input_EEF(pipeline_state0, action)
        else:
            arm_angles = self._get_arm_angles(pipeline_state0)
            action = self._convert_action_to_actuator_input_joint_angle(
                action, arm_angles, delta_control=self.config.action.delta
            )

        pipeline_state = self.pipeline_step(pipeline_state0, action)

        goal = state.info["goal"]
        goal_progress = state.info["goal_progress"]
        if self.nr_waypoints > 1:
            cube_pos = pipeline_state.q[:3]
            reached = (jnp.linalg.norm(cube_pos - goal[0]) < self.goal_radius) & (
                goal_progress < self.nr_waypoints
            )
            shifted_goal = jnp.concatenate([goal[1:], goal[-1:]], axis=0)
            goal = jnp.where(reached, shifted_goal, goal)
            goal_progress = goal_progress + reached.astype(jnp.int32)
            pipeline_state = self._update_goal_visualization(pipeline_state, goal)

        # ADDING THIS HERE BASED ON MICHAL'S BUG FIX
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        # Compute variables for state update, including observation and goal/reward
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, goal, timestep)

        final_goal = goal[-1] if self.nr_waypoints > 1 else goal
        success, success_easy, success_hard = self._compute_goal_completion(
            obs, final_goal
        )
        if self.nr_waypoints > 1:
            sequence_complete = (goal_progress >= self.nr_waypoints).astype(success.dtype)
            success = success * sequence_complete
            success_easy = success_easy * sequence_complete
            success_hard = success_hard * sequence_complete
        state.metrics.update(
            success=success, success_easy=success_easy, success_hard=success_hard
        )

        reward = success
        done = 0.0
        # info = {**state.info, "timestep": timestep}
        # ADDING THIS HERE BASED ON MICHAL'S BUG FIX
        info = {
            **state.info,
            "goal": goal,
            "goal_progress": goal_progress,
            "timestep": timestep,
            "seed": seed,
        }

        new_state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info
        )

        # Special postprocessing
        if (
            self.env_name == "arm_grasp"
        ):  # Update finger locations next to current cube location
            cube_pos = obs[:3]
            left_finger_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
            right_finger_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
            adjusted_goal = (
                state.info["goal"]
                .at[:6]
                .set(jnp.concatenate([left_finger_goal_pos] + [right_finger_goal_pos]))
            )
            new_state = self.update_goal(new_state, adjusted_goal)

        return new_state

    def update_goal(self, state: State, goal: jax.Array) -> State:
        """
        We deviate from the existing interface by allowing the goal to be modified directly,
        since this is useful for exploration during trajectory collection.

        We store the full goal virtually in state.info["goal"], rather than having it be a physical object
        in simulation (as we allow the goal space to be velocity, timestep, etc. which are not representable
        purely by position and orientation of objects).

        However, since the current goals for all environments each have some notion of goal position (end of
        arm, or cube location) as a subset of the goal, we opt to include these subsets physically in the simulation
        as uninteractable ghost objects for visualization purposes, stored in pipeline_state.q. We typically
        filter these out of the observation, however, as they are redundant with the already-included state.info["goal"].
        """

        info = {
            **state.info,
            "goal": goal,
        }  # Dictionary unpacking to return updated dict
        pipeline_state = self._update_goal_visualization(state.pipeline_state, goal)
        return state.replace(pipeline_state=pipeline_state, info=info)

    def _convert_action_to_actuator_input_joint_angle(
        self, action: jax.Array, arm_angles: jax.Array, delta_control=False
    ) -> jax.Array:
        """
        Converts the [-1, 1] actions to the corresponding target angle or gripper strength.
        We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.

        We restrict rotation to approximately the front two octants, and further restrict wrist rotation, to
        reduce the space of unreasonable actions. Hence the action dimension for the arm joints is 4 instead of 7,
        though the action given to the simulator itself needs to be 7 for the arm, plus 2 for the left and right fingers.

        delta_control: indicates whether or not to interpret the arm actions as targeting an offset from the current angle, or as
        targeting an absolute angle. Using delta control might improve convergence by reducing the effective action space at any timestep.
        """

        arm_action = jnp.array(
            [action[0], action[1], 0, action[2], 0, action[3], 0]
        )  # Expand to 4-dim to 7-dim, and fill in fixed values for joints 3, 5, 7
        min_value = jnp.array([0.3491, 0, 0, -3.0718, 0, 2.3562, 1.4487])
        max_value = jnp.array([2.7925, 1.48353, 0, -0.0698, 0, 3.7525, 1.4487])

        # If f(x) = offset + x * multiplier, then this offset and multiplier yield f(-1) = min_value, f(1) = max_value.
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2

        # Retrieve absolute angle target in [-1, 1] space from delta actions in [-1, 1]
        if delta_control:
            normalized_arm_angles = jnp.where(
                multiplier > 0, (arm_angles - offset) / multiplier, 0
            )  # Convert arm angles back to [-1, 1] space
            delta_range = self.config.action.max_delta  # If this number is 0.25, an action of +/- 1 targets an angle 25% of the max range away from the current angle.
            arm_action = normalized_arm_angles + arm_action * delta_range
            arm_action = jnp.clip(arm_action, -1, 1)

        # Rescale back to absolute angle space in radians
        arm_action = offset + arm_action * multiplier

        # Gripper control
        # Binary open-closedness: if positive, set to actuator value 0 (totally closed); if negative, set to actuator value 255 (totally open)
        if self.env_name not in ("arm_reach"):
            gripper_action = jnp.where(
                action[-1] > 0,
                jnp.array([0, 0], dtype=float),
                jnp.array([255, 255], dtype=float),
            )
            converted_action = jnp.concatenate([arm_action] + [gripper_action])
        else:
            converted_action = arm_action

        return converted_action

    def _convert_action_to_actuator_input_EEF(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        eef_index = 2
        current_position = pipeline_state.x.pos[eef_index]
        delta_range = 0.2  # Unlike arm angle control which is more complex, if this number is 0.2, an action of +/- 1 simply targets +/- 0.2 distance in position
        arm_action = current_position + delta_range * jnp.clip(action[:3], -1, 1)

        # Gripper control
        # Binary open-closedness: if positive, set to actuator value 0 (totally closed); if negative, set to actuator value 255 (totally open)
        gripper_action = jnp.where(
            action[-1] > 0,
            jnp.array([0, 0], dtype=float),
            jnp.array([255, 255], dtype=float),
        )

        converted_action = jnp.concatenate([arm_action] + [gripper_action])
        return converted_action

    # Methods to be overridden by specific environments
    def _get_xml_path(self):
        raise NotImplementedError

    def _set_environment_attributes(self):
        """
        Attribute descriptions:
        - Episode lengths are merely a recommendation + used for scaling timestep
        - Goal indices are for data collection/goal-conditioning; completion goal indices are for reward/goal completion checking
        - State_dim is for the observation dimension WITHOUT the goal appended; brax's env.observation_size gives the dimension WITH goal appended
        - Noise scale is for the range of uniform-distribution noise
            - Arm: perturb all joints (in radians) and gripper position (in meters)
            - Cube: perturb x, y of starting cube location
            - Goal: perturb x, y of cube destination or x, y, z of reach location
        """
        raise NotImplementedError

    def _get_initial_state(self, rng):
        raise NotImplementedError

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        raise NotImplementedError

    def _compute_goal_completion(self, obs, goal):
        raise NotImplementedError

    def _update_goal_visualization(
        self, pipeline_state: base.State, goal: jax.Array
    ) -> base.State:
        raise NotImplementedError

    def _get_obs(
        self, pipeline_state: base.State, goal: jax.Array, timestep
    ) -> jax.Array:
        raise NotImplementedError

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        raise NotImplementedError
