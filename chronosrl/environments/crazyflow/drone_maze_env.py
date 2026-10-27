"""Goal-conditioned Crazyflie maze environment on the crazyflow simulator."""

import jax
import jax.numpy as jnp
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace
from ml_collections import config_dict

from chronosrl.environments.crazyflow.drone_env import DroneEnv
from chronosrl.environments.maze_layouts import LAYOUTS
from chronosrl.environments.maze_layouts import G as GOAL
from chronosrl.environments.maze_layouts import R as RESET


class DroneMazeEnv(DroneEnv):
    """Fly one Crazyflie per world through a maze to a goal in a random free cell.

    The maze uses the ant maze layouts, with cell ``(i, j)`` centred at ``(i, j) * scaling``.
    Walls exist only in x and y: the episode terminates as soon as the drone is above a wall
    cell, so it can neither touch nor fly over them.
    """

    def __init__(self, config: config_dict.ConfigDict, nr_envs: int):
        """Build the base env and precompute the maze cells.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds, one drone each.
        """
        super().__init__(config, nr_envs)

        layout = LAYOUTS[config.maze.layout]
        s = self.scaling = config.maze.scaling
        self.walls = jnp.array([[c == 1 for c in row] for row in layout])  # (H, W) bool

        def cells(tag: object) -> jax.Array:
            # (n, 2) xy centres of the cells marked tag
            ij = [[i, j] for i, row in enumerate(layout) for j, c in enumerate(row) if c == tag]
            return jnp.array(ij, jnp.float32) * s

        self.start_xy, self.goal_xy = cells(RESET)[0], cells(GOAL)  # (2,), (n_goals, 2)
        self.wall_xy = cells(1)  # for rendering only

        # Drones spawn at the start cell, or at any free cell with random_start
        self.spawn_xy = self.start_xy[None]
        if config.maze.random_start:
            self.spawn_xy = jnp.concatenate([self.spawn_xy, self.goal_xy])
        self.spawn_noise = config.maze.spawn_noise  # m, max xy offset from a cell centre

    def in_wall(self, pos: jax.Array) -> jax.Array:
        """Return the ``(n,)`` worlds whose drone is inside a wall cell, checking only x and y."""
        idx = jnp.round(pos[:, :2] / self.scaling).astype(jnp.int32)
        idx = jnp.clip(idx, 0, jnp.array(self.walls.shape) - 1)
        return self.walls[idx[:, 0], idx[:, 1]]  # (n,) bool

    def terminated(self, pos: jax.Array, vel: jax.Array) -> jax.Array:
        """Base termination plus hitting a wall."""
        return super().terminated(pos, vel) | self.in_wall(pos)

    def evaluation_step_metrics(
        self, pos: jax.Array, vel: jax.Array, goal: jax.Array
    ) -> dict[str, jax.Array]:
        """Base metrics plus ``crashed``: the drone is inside a wall cell."""
        metrics = super().evaluation_step_metrics(pos, vel, goal)
        return {**metrics, "crashed": self.in_wall(pos).astype(jnp.float32)}

    def spawn(
        self, data: SimData, goal: jax.Array | None, key: jax.Array, mask: jax.Array | None = None
    ) -> tuple[SimData, jax.Array]:
        """Respawn the worlds in ``mask`` in a random spawn cell with a goal in a random goal cell.

        Positions and goals get a uniform xy offset of at most ``spawn_noise`` from the cell centre,
        with z sampled from the configured spawn and goal boxes. Velocity components are sampled
        uniformly up to ``max_spawn_speed``.

        Args:
            data: Simulator data of all worlds.
            goal: Current ``(n, 3)`` goals, kept where ``mask`` is False. Unused if no ``mask``.
            key: PRNG key.
            mask: ``(n,)`` boolean array of worlds to respawn, or None to respawn all of them.

        Returns:
            The updated simulator data and ``(n, 3)`` goals.
        """
        keys = jax.random.split(key, 7)
        n = self.nr_envs

        def sample(
            cell_key: jax.Array,
            noise_key: jax.Array,
            z_key: jax.Array,
            cells: jax.Array,
            z_low: jax.Array,
            z_high: jax.Array,
        ) -> jax.Array:
            # Random cell centre plus xy noise, z uniform in [z_low, z_high]
            xy = cells[jax.random.randint(cell_key, (n,), 0, len(cells))]
            xy = xy + jax.random.uniform(
                noise_key, (n, 2), minval=-self.spawn_noise, maxval=self.spawn_noise
            )
            z = jax.random.uniform(z_key, (n, 1), minval=z_low, maxval=z_high)
            return jnp.concatenate([xy, z], axis=-1)  # (n, 3)

        pos = sample(*keys[:3], self.spawn_xy, self.spawn_low[2], self.spawn_high[2])[:, None]
        new_goal = sample(*keys[3:6], self.goal_xy, self.goal_low[2], self.goal_high[2])
        vel = jax.random.uniform(
            keys[6], (n, 1, 3), minval=-self.max_spawn_speed, maxval=self.max_spawn_speed
        )

        data = self.sim._reset(data, self.sim.default_data, mask)
        spawn_state = {"pos": pos, "vel": vel}
        if self.reset_rotor:
            spawn_state["rotor_vel"] = jnp.full((self.nr_envs, 1, 4), self.hover_rotor_vel)
        data = data.replace(states=leaf_replace(data.states, mask, **spawn_state))

        goal = new_goal if mask is None else jnp.where(mask[:, None], new_goal, goal)

        return data, goal

    def policy_observation(self, obs: jax.Array) -> jax.Array:
        """Turn ``(..., 21)`` observations into ``(..., 21)`` policy inputs."""
        pos = obs[..., :3]
        rot_mat = obs[..., 3:12]
        vel = obs[..., 12:15]
        goal = obs[..., self.state_dim :]
        # pos_to_goal_body = rotation.apply(goal - pos, inverse=True)
        # vel_body = rotation.apply(vel, inverse=True)
        ang_vel = obs[..., 15 : self.state_dim]  # is already in body frame

        return jnp.concatenate([pos, self.scaling * (pos - goal), vel, rot_mat, ang_vel], axis=-1)

    def observation(self, data: SimData, goal: jax.Array) -> jax.Array:
        """Base obs with position and goal in maze units."""
        obs = super().observation(data, goal)
        obs = obs.at[..., :3].divide(self.scaling)

        return obs.at[..., self.state_dim :].divide(self.scaling)
