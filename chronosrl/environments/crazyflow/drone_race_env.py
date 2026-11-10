"""Goal-conditioned Crazyflie racing environment through the gates of an lsy drone racing track.

The track loading and contact check follow ``lsy_drone_racing.envs.race_core``, the gate and
obstacle models in ``assets/`` are copied from ``lsy_drone_racing/envs/assets``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from crazyflow.sim.data import SimData
from crazyflow.sim.sim import sync_sim2mjx, use_box_collision
from crazyflow.utils import leaf_replace
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import config_dict
from mujoco.mjx import Data
from scipy.spatial.transform import Rotation

from chronosrl.environments.crazyflow.drone_env import DroneEnv, State

ASSETS = Path(__file__).parent / "assets"


class DroneRaceEnv(DroneEnv):
    """Fly through the gates of a track in ``gate_order``, terminating on contact with the track.

    Every gate pass in ``gate_order`` is three waypoints along the gate's normal (its local +x,
    the lsy passing direction): an approach point ``waypoint_offset`` in front of the gate, the
    centre and an exit point ``waypoint_offset`` behind it, so the drone crosses each gate along
    its normal before turning. Each episode the gates and obstacles are moved by uniform noise
    around their nominal poses, and the sampled track is the ``task``.
    """

    def __init__(self, config: config_dict.ConfigDict, nr_envs: int):
        """Build the base env, add the gates and obstacles to the sim and precompute the contacts.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds, one drone each.
        """
        # Before super(): the base sizes the observation from it
        self.state_dim = DroneEnv.state_dim + 6 * len(config.track.gates)
        super().__init__(config, nr_envs)
        track = config.track
        assert all(0 < g <= len(track.gates) for g in track.gate_order), "gate numbers are 1..G"
        self.nr_passes = len(track.gate_order)
        self.centres_only = self.nr_waypoints == self.nr_passes  # no approach and exit points
        assert self.centres_only or self.nr_waypoints == 3 * self.nr_passes, (
            "one or three waypoints per gate_order entry"
        )
        self.waypoint_offset = track.get("waypoint_offset", 0.4)  # m, approach and exit distance

        self.nominal_gates_pos = jnp.array([g["pos"] for g in track.gates])  # (G, 3)
        self.nominal_gates_rpy = jnp.array([g["rpy"] for g in track.gates])  # (G, 3)
        self.nominal_obstacles_pos = jnp.array([o["pos"] for o in track.obstacles])  # (O, 3)
        self.gate_sequence = jnp.array(track.gate_order) - 1  # 1-based gate numbers to ids
        # Spawn in the gates too, not only on the platform (see sample_task)
        self.spawn_at_gates = config.task.get("spawn_at_gates", False)
        # With spawn_at_gates, the share of worlds that still start on the platform
        self.platform_probability = config.task.get("platform_probability", 0.5)
        # m/s, gate spawns start with a uniform speed up to this along the gate normal
        self.spawn_speed = config.task.get("spawn_speed", 0.0)

        # Uniform noise half-widths per axis
        noise = track.noise
        self.gate_pos_noise = jnp.array(noise.gate_pos)
        self.gate_rpy_noise = jnp.array(noise.gate_rpy)
        self.obstacle_pos_noise = jnp.array(noise.obstacle_pos)

        # Order matters: the contact mask and the mocap ids need the final mjx model
        self.load_track_into_sim(track)
        use_box_collision(self.sim, True)  # rebuilds the mjx model
        self.contact_mask = self.drone_contact_mask()
        model = self.sim.mj_model
        self.gate_ids = [int(model.body(f"gate:{i}").mocapid[0]) for i in range(len(track.gates))]
        self.obstacle_ids = [
            int(model.body(f"obstacle:{i}").mocapid[0]) for i in range(len(track.obstacles))
        ]

    def gate_features(self, task: dict) -> jax.Array:
        """Return the (n, G, 6) centre and normal of each gate."""
        n, G = task["gates_pos"].shape[:2]
        normal = (
            R.from_quat(task["gates_quat"].reshape(-1, 4)).as_matrix()[:, :, 0].reshape(n, G, 3)
        )
        return jnp.concatenate([task["gates_pos"], normal], axis=-1)

    def add_gates(self, obs: jax.Array, task: dict) -> jax.Array:
        """Insert the gate features between the drone state and the goals of a base ``obs``."""
        n = DroneEnv.state_dim
        gates = self.gate_features(task).reshape(obs.shape[0], -1).astype(obs.dtype)
        return jnp.concatenate([obs[:, :n], gates, obs[:, n:]], axis=-1)

    def reset(self, keys: jax.Array) -> State:
        """Base reset with the gates in the observation."""
        state = super().reset(keys)
        return state.replace(obs=self.add_gates(state.obs, state.task))

    def step(self, state: State, action: jax.Array) -> State:
        """Base step with the gates in the observation."""
        next_state = super().step(state, action)
        return next_state.replace(obs=self.add_gates(next_state.obs, next_state.task))

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Goals and gate centres relative to the drone, gate normals, rotation and velocities."""
        n, batch = DroneEnv.state_dim, obs.shape[:-1]
        pos = obs[..., None, :3]
        gates = obs[..., n : self.state_dim].reshape(batch + (-1, 6))
        goals = obs[..., self.state_dim :].reshape(batch + (self.nr_waypoints, 3))
        return jnp.concatenate(
            [
                (pos - goals).reshape(batch + (-1,)),
                obs[..., 3:n],
                (pos - gates[..., :3]).reshape(batch + (-1,)),
                gates[..., 3:].reshape(batch + (-1,)),
            ],
            axis=-1,
        )

    def gate_waypoints(self, task: dict) -> jax.Array:
        """Return the ``(n, P, 3, 3)`` approach, centre and exit points of the P gate passes."""
        centre = task["gates_pos"][:, self.gate_sequence]  # (n, P, 3)
        quat = task["gates_quat"][:, self.gate_sequence].reshape(-1, 4)
        normal = R.from_quat(quat).apply(jnp.array([1.0, 0.0, 0.0])).reshape(centre.shape)
        offset = self.waypoint_offset * normal
        return jnp.stack([centre - offset, centre, centre + offset], axis=2)

    def sample_task(self, key: jax.Array) -> tuple[jax.Array, dict]:
        """Sample a noisy track and a start per world; the goals are the waypoints left from it.

        ``start = 0`` spawns on the platform with all waypoints ahead, ``j > 0`` spawns in the
        centre of gate pass ``j - 1`` with its exit point and the passes ``j..`` ahead, padded with
        the last waypoint. With ``spawn_at_gates``, a world starts on the platform with
        ``platform_probability`` and otherwise in one of the passes ``0..P-2`` uniformly; without
        it every world starts on the platform.

        Returns:
            ``(n, K, 3)`` goals and the task: ``gates_pos`` ``(n, G, 3)``, ``gates_quat``
            ``(n, G, 4)`` as xyzw, ``obstacles_pos`` ``(n, O, 3)`` and ``start`` ``(n,)``.
        """
        pos_key, rpy_key, obstacle_key, start_key = jax.random.split(key, 4)

        def perturb(key: jax.Array, nominal: jax.Array, half_width: jax.Array) -> jax.Array:
            # Nominal plus uniform noise in [-half_width, half_width], (n, *nominal.shape)
            shape = (self.nr_envs, *nominal.shape)
            return nominal + jax.random.uniform(key, shape, minval=-half_width, maxval=half_width)

        gates_pos = perturb(pos_key, self.nominal_gates_pos, self.gate_pos_noise)
        gates_rpy = perturb(rpy_key, self.nominal_gates_rpy, self.gate_rpy_noise)
        gates_quat = R.from_euler("xyz", gates_rpy.reshape(-1, 3)).as_quat()
        obstacles_pos = perturb(obstacle_key, self.nominal_obstacles_pos, self.obstacle_pos_noise)

        if self.spawn_at_gates:
            platform_key, gate_key = jax.random.split(start_key)
            on_platform = jax.random.bernoulli(
                platform_key, self.platform_probability, (self.nr_envs,)
            )
            in_gate = jax.random.randint(gate_key, (self.nr_envs,), 1, self.nr_passes + 1)
            start = jnp.where(on_platform, 0, in_gate)
        else:
            start = jnp.zeros(self.nr_envs, jnp.int32)
        task = {
            "gates_pos": gates_pos,
            "gates_quat": gates_quat.reshape(self.nr_envs, -1, 4),
            "obstacles_pos": obstacles_pos,
            "start": start,
        }
        # Indexed by gate_order, so a gate passed twice is the same gate both times
        K = self.nr_waypoints
        waypoints = self.gate_waypoints(task)
        if self.centres_only:
            waypoints = waypoints[:, :, 1:2]
        waypoints = waypoints.reshape(self.nr_envs, K, 3)
        # A spawn in pass j - 1 sits at its centre, so the goals start at its exit point, or at
        # the next centre without exit points
        first = jnp.where(start > 0, start -1 if self.centres_only else 3 * start - 2, 0)
        idx = jnp.minimum(jnp.arange(K)[None] + first[:, None], K - 1)  # (n, K)
        goals = waypoints[jnp.arange(self.nr_envs)[:, None], idx]  # remaining waypoints, padded
        return goals, task

    def spawn(
        self,
        data: SimData,
        goal: jax.Array | None,
        task: dict | None,
        key: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[SimData, jax.Array, dict]:
        """Base spawn, then move the worlds with ``start > 0`` to the centre of pass ``start - 1``.

        The base spawn samples the platform position from the spawn box and merges the new task
        by ``mask``, so ``task["start"]`` already belongs to the new episodes.
        """
        key, speed_key = jax.random.split(key)
        data, goal, task = super().spawn(data, goal, task, key, mask)
        waypoints = self.gate_waypoints(task)  # (n, P, 3, 3)
        rows, pass_id = jnp.arange(self.nr_envs), jnp.maximum(task["start"] - 1, 0)
        in_gate = waypoints[rows, pass_id, 0]
        normal = (waypoints[rows, pass_id, 1] - in_gate) / self.waypoint_offset
        speed = jax.random.uniform(speed_key, (self.nr_envs, 1), maxval=self.spawn_speed)

        at_gate = (task["start"] > 0)[:, None]
        pos = jnp.where(at_gate, in_gate, data.states.pos[:, 0])
        vel = jnp.where(at_gate, speed * normal, data.states.vel[:, 0])
        data = data.replace(
            states=leaf_replace(data.states, mask, pos=pos[:, None], vel=vel[:, None])
        )
        return data, goal, task

    def place_track(self, mjx_data: Data, task: dict) -> Data:
        """Move the gate and obstacle mocap bodies of ``mjx_data`` to the track in ``task``."""
        pos = mjx_data.mocap_pos.at[:, self.gate_ids].set(task["gates_pos"])
        pos = pos.at[:, self.obstacle_ids].set(task["obstacles_pos"])
        # MuJoCo quaternions are wxyz
        quat = mjx_data.mocap_quat.at[:, self.gate_ids].set(jnp.roll(task["gates_quat"], 1, -1))
        return mjx_data.replace(mocap_pos=pos, mocap_quat=quat)

    def crashed(self, data: SimData, task: dict) -> jax.Array:
        """Return the ``(n,)`` worlds whose drone touches a gate or an obstacle."""
        mjx_data = self.place_track(self.sim.mjx_data, task)  # captured as a constant, like lsy
        _, mjx_data = sync_sim2mjx(data, mjx_data, self.sim.mjx_model)
        return jnp.any((mjx_data._impl.contact.dist < 0) & self.contact_mask, axis=-1)

    def terminated(self, data: SimData, task: dict) -> jax.Array:
        """Base termination plus touching a gate or an obstacle."""
        return super().terminated(data, task) | self.crashed(data, task)

    def evaluation_step_metrics(
        self, pos: jax.Array, vel: jax.Array, goal: jax.Array, task: dict
    ) -> dict[str, jax.Array]:
        """Base metrics plus ``waypoint_distance``: ``(n, P)`` distances to every gate centre.

        The distances are to the centres of all P gate passes, also the ones already passed; the
        approach and exit points are left out. ``EvalWrapper`` keeps their minimum over the
        episode, the closest approach to each gate.
        """
        metrics = super().evaluation_step_metrics(pos, vel, goal, task)
        centres = task["gates_pos"][:, self.gate_sequence]  # (n, P, 3)
        distance = jnp.linalg.norm(centres - pos[:, None], axis=-1)
        return {**metrics, "waypoint_distance": distance}

    def load_track_into_sim(self, track: config_dict.ConfigDict):
        """Attach the nominal gates and obstacles to the sim as mocap bodies."""
        gate_body = mujoco.MjSpec.from_file(str(ASSETS / "gate.xml")).body("gate")
        obstacle_body = mujoco.MjSpec.from_file(str(ASSETS / "obstacle.xml")).body("obstacle")
        frame = self.sim.spec.worldbody.add_frame()
        for i, g in enumerate(track.gates):
            gate = frame.attach_body(gate_body, "", f":{i}")
            gate.pos = g["pos"]
            gate.quat = Rotation.from_euler("xyz", g["rpy"]).as_quat(scalar_first=True)  # wxyz
            gate.mocap = True  # mocap bodies can be moved per world
        for i, o in enumerate(track.obstacles):
            obstacle = frame.attach_body(obstacle_body, "", f":{i}")
            obstacle.pos = o["pos"]
            obstacle.mocap = True
        # No build_mjx here: use_box_collision rebuilds the mjx model afterwards

    def drone_contact_mask(self) -> np.ndarray:
        """Return the ``(n_contacts,)`` mask of contacts with the drone but not the floor."""
        sim = self.sim
        sim.contacts()  # syncs mjx, so the contact buffer is filled
        contact = sim.mjx_data._impl.contact
        # The contact layout is the same in every world, so world 0 is enough
        geom1, geom2 = np.asarray(contact.geom1[0]), np.asarray(contact.geom2[0])

        def involves(body: str) -> np.ndarray:
            body_id = sim.mj_model.body(body).id
            start = sim.mj_model.body_geomadr[body_id]
            end = start + sim.mj_model.body_geomnum[body_id]
            return ((geom1 >= start) & (geom1 < end)) | ((geom2 >= start) & (geom2 < end))

        return involves("drone:0") & ~involves("world")
