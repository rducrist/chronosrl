from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from chronosrl.environments.jaxgcrl.manipulation.arm_envs import ArmEnvs, asset_path

"""
Binpick-Hard: Lift a cube from the blue bin, then move it to a random location in the red bin.
- Observation space: 17-dim obs + two 3D cube-position goals.
- Action space:      5-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""


class ArmBinpickHard(ArmEnvs):
    def _get_xml_path(self):
        return asset_path("panda_binpick_hard.xml")

    @property
    def action_size(self) -> int:
        return 5  # Override default (actuator count)

    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_binpick_hard"
        self.episode_length = self.config.episode_length

        self.goal_indices = (0, 3)  # Cube position
        self.goal_radius = self.config.task.goal_radius  # Cube position
        self.completion_goal_indices = jnp.array([0, 1, 2])  # Identical
        self.state_dim = 17
        self.nr_waypoints = 2
        self.lift_height = self.config.task.lift_height

        self.arm_noise_scale = self.config.spawn.arm_noise
        self.cube_noise_scale = self.config.spawn.cube_noise
        self.goal_noise_scale = self.config.goal.noise

    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(
            subkey1, [2], minval=-1
        )
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array(self.config.spawn.arm_q)  # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(
            subkey2, [self.sys.q_size() - 14], minval=-1
        )

        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array(self.config.goal.pos) + jnp.array(
            [self.goal_noise_scale, self.goal_noise_scale, 0]
        ) * jax.random.uniform(subkey, [3], minval=-1)
        lift_goal = pipeline_state.q[:3].at[2].add(self.lift_height)
        return jnp.stack([lift_goal, cube_goal_pos])

    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the cube close enough to the goal?
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[-1] if goal.ndim > 1 else goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)

        task = self.config.task
        success = jnp.array(dist < self.goal_radius, dtype=float)
        success_easy = jnp.array(dist < task.easy_goal_radius, dtype=float)
        success_hard = jnp.array(dist < task.hard_goal_radius, dtype=float)
        return success, success_easy, success_hard

    def _update_goal_visualization(
        self, pipeline_state: base.State, goal: jax.Array
    ) -> base.State:
        goal_pos = goal.reshape(-1)[:3]
        updated_q = pipeline_state.q.at[7:10].set(goal_pos)  # Only set the position, not orientation
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state

    def _get_obs(
        self, pipeline_state: base.State, goal: jax.Array, timestep
    ) -> jax.Array:
        """
        Observation space (17-dim)
         - q_subset (10-dim): 3-dim cube position, 7-dim joint angles
         - End-effector (6-dim): position and velocity
         - Fingers (1-dim): finger distance
        Note q is 23-dim: 7-dim cube position/angle, 7-dim goal marker position/angle, 7-dim joint angles, 2-dim finger offset

        Goal space (3-dim): position of cube
        """
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]

        eef_index = 8  # Cube is 0, goal marker is 1, then links 1-7 are indices 2-8. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        left_finger_index = 9
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 10
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[
            None
        ]  # [None] expands dims from 0 to 1

        return jnp.concatenate(
            [q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [goal.reshape(-1)]
        )

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        return obs

    def training_metrics(self, state: State) -> dict:
        obs = state.obs
        cube, eef = obs[:, 0:3], obs[:, 10:13]
        goal = obs[:, self.state_dim:].reshape((-1, self.nr_waypoints, 3))
        cube_goal = jnp.linalg.norm(cube - goal[:, -1], axis=-1)
        return {
            "train/success": jnp.mean(state.metrics["success"]),
            "train/success_hard": jnp.mean(state.metrics["success_hard"]),
            "train/cube_goal_distance": jnp.mean(cube_goal),
            "train/lift_waypoint_reached": jnp.mean(state.info["goal_progress"] > 0),
            # is the arm reaching the cube at all? the first thing to improve
            "train/eef_cube_distance": jnp.mean(jnp.linalg.norm(eef - cube, axis=-1)),
            # binpick must lift the cube out of the blue bin, where it spawns at z ~ 0.07
            "train/cube_height": jnp.mean(cube[:, 2]),
            "train/gripper_closed": jnp.mean(obs[:, 16] < 0.04),
        }
