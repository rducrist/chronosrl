import jax.numpy as jnp
from mujoco import mjx
from brax import envs

from chronosrl.environments.jaxgcrl.ant_maze import AntMaze
from chronosrl.environments.jaxgcrl.humanoid import Humanoid, HumanoidMaze
from chronosrl.environments.jaxgcrl.manipulation.arm_push_easy import ArmPushEasy


# brax 0.10.1 still calls mjx.ncon, which MuJoCo 3.2 removed
if not hasattr(mjx, "ncon"):
    mjx.ncon = lambda sys: mjx.make_data(sys).ncon

TASKS = {
    "ant_u4_maze": lambda config: AntMaze("u4_maze"),
    "ant_big_maze": lambda config: AntMaze("big_maze"),
    "ant_hardest_maze": lambda config: AntMaze("hardest_maze"),
    "ant_u5_maze": lambda config: AntMaze("u5_maze"),
    "humanoid": lambda config: Humanoid(),
    "humanoid_u_maze": lambda config: HumanoidMaze("u_maze"),
    "humanoid_big_maze": lambda config: HumanoidMaze("big_maze"),
    "arm_push_easy": lambda config: ArmPushEasy(config),
}


class EvalWrapper(envs.training.EvalWrapper):
    def evaluation_metrics(self, state):
        metrics = state.info["eval_metrics"]
        success = metrics.episode_metrics["success"]
        return {"time_at_goal": jnp.mean(success), "goal_reached": jnp.mean(success > 0), "episode_length": jnp.mean(metrics.episode_steps)}


# Hyperparameters live in configs/<environment_name>.toml
def eval_config(config):
    """Copy of the env ``config`` with ``task.eval_goal_radius`` in place of ``task.goal_radius``."""
    config = config.copy_and_resolve_references()
    if "task" in config and "eval_goal_radius" in config.task:
        config.task.goal_radius = config.task.eval_goal_radius
    return config


def create_env(config):
    env_config = config.environment
    train_env = envs.training.wrap(TASKS[env_config.name](env_config), episode_length=env_config.episode_length)
    eval_env = EvalWrapper(envs.training.wrap(TASKS[env_config.name](eval_config(env_config)), episode_length=env_config.episode_length))
    return train_env, eval_env
