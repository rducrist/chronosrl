"""Goal-conditioned Crazyflie environment that flies a fixed track of noisy nominal waypoints."""

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.environments.crazyflow.drone_env import DroneEnv


class DroneTrackEnv(DroneEnv):
    """Fly a fixed track: each waypoint is a nominal point plus truncated normal noise."""

    def __init__(self, config: config_dict.ConfigDict, nr_envs: int):
        """Build the base env and store the nominal waypoints and their noise.

        Args:
            config: The ``environment`` section of the experiment config.
            nr_envs: Number of parallel worlds, one drone each.
        """
        super().__init__(config, nr_envs)

        self.goal_nominal = jnp.array(config.goal.nominal)
        self.goal_std = jnp.array(config.goal.std)
        self.goal_clip = config.goal.clip
        assert self.goal_nominal.shape == (self.nr_waypoints, 3), "one nominal row per waypoint"

    def sample_task(self, key: jax.Array) -> tuple[jax.Array, dict]:
        """Sample ``(n, K, 3)`` goals: the nominal waypoints plus noise clipped at ``clip`` std."""
        noise = jax.random.truncated_normal(
            key, -self.goal_clip, self.goal_clip, (self.nr_envs, self.nr_waypoints, 3)
        )

        return self.goal_nominal + self.goal_std * noise, {}
