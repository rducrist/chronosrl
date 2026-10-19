import jax.numpy as jnp

from chronosrl.environments.go2.goal_environment import Go2GoalEnv


class EvalWrapper:
    def __init__(self, env):
        self.env = env


    def __getattr__(self, name):
        return getattr(self.env, name)


    def reset(self, keys):
        state = self.env.reset(keys)
        return state.replace(info={**state.info, "episode_metrics": {name: jnp.zeros_like(value) for name, value in state.metrics.items()},
                                   "active_episodes": jnp.ones_like(state.done), "episode_steps": jnp.zeros_like(state.done)})


    def step(self, state, action):
        next_state = self.env.step(state, action)
        active = state.info["active_episodes"]
        return next_state.replace(info={
            **next_state.info,
            "episode_metrics": {name: state.info["episode_metrics"][name] + value * active for name, value in next_state.metrics.items()},
            "active_episodes": active * (1.0 - next_state.done),
            "episode_steps": jnp.where(active, next_state.info["steps"], state.info["episode_steps"]),
        })


    def evaluation_metrics(self, state):
        episode_metrics = {name: jnp.mean(value) for name, value in state.info["episode_metrics"].items()}
        metrics = {"time_at_goal": episode_metrics["success"], "episode_length": jnp.mean(state.info["episode_steps"])}
        if self.env.task == "velocity":
            metrics["forward_time_at_goal"] = episode_metrics["forward_success"]
            metrics["forward_tracking_error"] = episode_metrics["forward_error"] / jnp.maximum(episode_metrics["forward_steps"], 1.0)
            metrics["forward_episode_length"] = episode_metrics["forward_steps"]
        return metrics


# Hyperparameters live in configs/<environment_name>.toml


def create_env(config):
    env_config = config.environment
    task = env_config.name.split("_")[1]
    make = lambda nr_envs, seed, is_eval: Go2GoalEnv(task, nr_envs, seed, is_eval, env_config.gait_amplitude, env_config.gait_frequency, env_config.travel_cost_weight)
    return make(env_config.nr_envs, env_config.seed, False), EvalWrapper(make(env_config.nr_eval_envs, env_config.seed + 1, True))
