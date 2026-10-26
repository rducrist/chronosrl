"""Evaluate a crazyflow_position policy from a handover hover, as after a scripted take-off.

The policy is trained with random spawns and goals. Here every drone starts at a fixed handover
point with a small leftover velocity and chases goals from the deployment range, so the metrics
tell whether a take-off controller can hand over to the policy. Afterwards a few episodes are
replayed in Crazyflow's MuJoCo viewer.

python -m chronosrl.evaluation.crazyflow.drone_reach_pos --run runs/chronosrl/crazyflow_position
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from chronosrl.environments.crazyflow import CrazyflowPositionEnv, EvalWrapper, State
from chronosrl.networks import Policy


def handover_config(args: argparse.Namespace, saved: dict) -> config_dict.ConfigDict:
    """Return the run's environment config with spawn and goals replaced by the handover setup."""
    env_config = config_dict.ConfigDict(saved)
    env_config.device = args.device
    env_config.spawn.low = env_config.spawn.high = args.handover
    env_config.spawn.max_speed = args.handover_speed
    env_config.goal.low = args.goal_low
    env_config.goal.high = args.goal_high
    return env_config


def evaluate(
    env_config: config_dict.ConfigDict, policy: Policy, params: Any, nr_worlds: int, seed: int
) -> dict[str, float]:
    """Run one episode in each of ``nr_worlds`` worlds and return the averaged eval metrics."""
    env = EvalWrapper(CrazyflowPositionEnv(env_config, nr_worlds))

    @jax.jit
    def run(key: jax.Array) -> dict[str, jax.Array]:
        def body(state: State, _: None) -> tuple[State, None]:
            action = jnp.tanh(policy.apply(params, env.policy_observation(state.obs))[0])
            return env.step(state, action), None

        state = env.reset(jax.random.split(key, nr_worlds))
        state, _ = jax.lax.scan(body, state, None, length=env.episode_length)
        return env.evaluation_metrics(state)

    metrics = {name: float(value) for name, value in run(jax.random.PRNGKey(seed)).items()}
    # Out of bounds is the only termination, so episodes shorter than the limit crashed
    metrics["max_episode_length"] = env.episode_length
    env.sim.close()
    return metrics


def view(
    env_config: config_dict.ConfigDict, policy: Policy, params: Any, args: argparse.Namespace
) -> None:
    """Roll out ``args.episodes`` consecutive episodes in one world and replay them in MuJoCo."""
    import glfw
    import mujoco

    env = CrazyflowPositionEnv(env_config, nr_envs=1)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    act = jax.jit(lambda obs: jnp.tanh(policy.apply(params, env.policy_observation(obs))[0]))

    # Autoreset starts the next episode in the same world, so the frames run back to back
    state = reset(jax.random.split(jax.random.PRNGKey(args.seed), 1))
    frames = [(state.data, np.asarray(state.goal[0]))]
    episodes = 0
    while episodes < args.episodes:
        state = step(state, act(state.obs))
        frames.append((state.data, np.asarray(state.goal[0])))
        episodes += int(state.done[0])

    dt = 1.0 / env_config.freq
    frame_count = int(np.ceil((len(frames) - 1) * dt * args.fps)) + 1
    print("Opening Crazyflow's native MuJoCo viewer (red marker = target).", flush=True)
    try:
        while True:
            started = time.perf_counter()
            for index in range(frame_count):
                viewer = env.sim.viewer
                if viewer is not None:
                    window = viewer.viewer.window
                    if window is None or glfw.window_should_close(window):
                        return
                data, goal = frames[min(int(index / (args.fps * dt)), len(frames) - 1)]
                env.sim.data = data
                if viewer is not None:
                    # The viewer clears markers after every render, so re-add the goal each frame.
                    viewer.viewer.add_marker(
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        pos=goal,
                        size=np.full(3, env.goal_radius),
                        rgba=np.array([1.0, 0.0, 0.0, 0.5]),
                        label="",
                    )
                env.sim.render(world=0)
                time.sleep(max(0.0, started + (index + 1) / args.fps - time.perf_counter()))
            if not args.loop:
                break
    finally:
        env.sim.close()


def main():
    """Parse arguments, evaluate the handover setup and replay a few episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory or its parent")
    parser.add_argument(
        "--epoch", type=int, help="Load checkpoints/epoch_<N>.pkl instead of the latest model.pkl"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nr-worlds", type=int, default=1000)
    parser.add_argument(
        "--handover", type=float, nargs=3, default=[0.0, 0.0, 0.5], help="Handover position xyz"
    )
    parser.add_argument(
        "--handover-speed", type=float, default=0.1, help="Max leftover speed per axis at handover"
    )
    parser.add_argument("--goal-low", type=float, nargs=3, default=[-0.5, -0.5, 0.3])
    parser.add_argument("--goal-high", type=float, nargs=3, default=[0.5, 0.5, 1.0])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes shown in the viewer")
    parser.add_argument("--fps", type=float, default=60, help="Playback frames per second")
    parser.add_argument("--no-render", action="store_true", help="Only print the metrics")
    parser.add_argument(
        "--loop", action="store_true", help="Repeat playback until the window is closed"
    )
    args = parser.parse_args()

    run = args.run.resolve()
    # A parent directory is also accepted: use its last run (by name) with a model.pkl
    if not (run / "model.pkl").exists():
        candidates = sorted(run.glob("*/model.pkl"), key=lambda p: p.parent.name)
        if not candidates:
            parser.error(f"No model.pkl found in {run}")
        run = candidates[-1].parent
    checkpoint_path = run / "model.pkl"
    if args.epoch is not None:
        checkpoint_path = run / "checkpoints" / f"epoch_{args.epoch:03d}.pkl"
        if not checkpoint_path.exists():
            parser.error(f"No checkpoint for epoch {args.epoch}: {checkpoint_path}")

    config = json.loads((run / "config.json").read_text())
    if config["environment"]["name"] != "crazyflow_position":
        parser.error(f"{run} is a {config['environment']['name']} run, not crazyflow_position")
    algorithm = config["algorithm"]
    if algorithm["action_chunk_length"] != 1:
        parser.error("Only action_chunk_length = 1 is supported")
    with checkpoint_path.open("rb") as file:
        params = pickle.load(file)["policy"]
    policy = Policy(CrazyflowPositionEnv.action_size, algorithm["width"], algorithm["depth"])
    env_config = handover_config(args, config["environment"])

    print(f"Evaluating {checkpoint_path} from handover {args.handover}", flush=True)
    metrics = evaluate(env_config, policy, params, args.nr_worlds, args.seed)
    print(json.dumps(metrics, indent=2), flush=True)

    if not args.no_render:
        # A single world renders fine on the CPU
        env_config.device = "cpu"
        view(env_config, policy, params, args)


if __name__ == "__main__":
    main()
