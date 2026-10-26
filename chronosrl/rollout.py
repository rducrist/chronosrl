"""Roll out saved policies: Crazyflow native viewer, pendulum animation or JaxGCRL HTML playback.

python -m chronosrl.rollout --run runs/chronosrl/crazyflow_position
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from chronosrl.environments import get_environment
from chronosrl.networks import Policy


def rollout_crazyflow(
    args: argparse.Namespace, run: Path, config: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    """Roll out a native crazyflow checkpoint, save metrics and replay it in MuJoCo.

    Args:
        args: Parsed command-line arguments of `main`.
        run: Run directory containing `model.pkl`.
        config: Saved run config with "environment" and "algorithm" sections.
        checkpoint: Unpickled `model.pkl`; its "policy" entry holds the policy params.
    """
    import mujoco
    from ml_collections import config_dict

    from chronosrl.environments.crazyflow import ENVS

    env_config = config_dict.ConfigDict(config["environment"])
    env_config.device = args.device
    if "maze" in env_config:
        env_config.maze.random_start = False  # always start in the R cell

    env = ENVS[env_config.name](env_config, nr_envs=1)
    dt = 1.0 / env_config.freq
    algorithm = config["algorithm"]
    chunk_length = algorithm["action_chunk_length"]
    policy = Policy(env.action_size * chunk_length, algorithm["width"], algorithm["depth"])
    params = checkpoint["policy"]
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    # Deterministic policy: tanh of the mean, a flat chunk of `chunk_length` actions.
    act = jax.jit(lambda obs: jnp.tanh(policy.apply(params, env.policy_observation(obs))[0]))
    print(
        f"Rolling out {run.name} ({env_config.name}), seed {args.seed}, device {args.device}",
        flush=True,
    )
    state = reset(jax.random.split(jax.random.PRNGKey(args.seed), 1))
    # Each frame is the sim data plus the goal the drone is chasing.
    frames = [(state.data, np.asarray(state.goal[0]))]
    successes, errors = [], []
    ended = False
    for index in range(args.steps):
        # Query the policy once per chunk and execute the chunk's actions open-loop.
        if index % chunk_length == 0:
            chunk = act(state.obs).reshape(1, chunk_length, env.action_size)
        state = step(state, chunk[:, index % chunk_length])
        successes.append(float(state.metrics["success"][0]))
        errors.append(float(state.metrics["goal_error"][0]))
        if bool(state.done[0]):
            # Same-step autoreset returns a different episode: omit that frame.
            ended = True
            break
        frames.append((state.data, np.asarray(state.goal[0])))
    output = args.output or run / f"rollout{epoch_suffix(args)}_seed{args.seed}.json"
    if output.suffix != ".json":
        raise ValueError(
            "Crazyflow --output must be a .json metrics file; playback uses its native viewer"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run": str(run),
        "environment": env_config.name,
        "seed": args.seed,
        "steps": len(successes),
        "time_at_goal": float(np.sum(successes)),
        "goal_reached": bool(np.any(successes)),
        "goal_error": float(np.mean(errors)),
        "episode_ended": ended,
    }
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Metrics: {output.resolve()}", flush=True)
    if args.no_render:
        env.sim.close()
        return
    print("Opening Crazyflow's native MuJoCo viewer (red marker = target).", flush=True)
    # Maze envs: walls exist only as xy cells, drawn as boxes up to the ceiling of the bounds
    wall_xy = np.asarray(getattr(env, "wall_xy", np.zeros((0, 2))))
    cam_config = None
    if len(wall_xy):
        wall_height = float(env.bounds_high[2])
        wall_size = np.array([env.scaling / 2, env.scaling / 2, wall_height / 2])
        # Look down on the maze centre from far enough to see all of it
        rows, cols = env.walls.shape
        center = np.array([(rows - 1) * env.scaling / 2, (cols - 1) * env.scaling / 2, 0.0])
        cam_config = {
            "lookat": center,
            "distance": 1.2 * max(rows, cols) * env.scaling,
            "elevation": -60.0,
        }
    try:
        import glfw

        frame_count = int(np.ceil((len(frames) - 1) * dt * args.fps)) + 1
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
                    for x, y in wall_xy:
                        viewer.viewer.add_marker(
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=np.array([x, y, wall_size[2]]),
                            size=wall_size,
                            rgba=np.array([0.7, 0.5, 0.3, 0.5]),
                            label="",
                        )
                env.sim.render(world=0, cam_config=cam_config)
                time.sleep(max(0.0, started + (index + 1) / args.fps - time.perf_counter()))
            if not args.loop:
                break
    finally:
        env.sim.close()


def pendulum_frame(
    theta: float, goal: float, goal_radius: float, torque: float, step: int, success: bool
) -> np.ndarray:
    """Draw one pendulum frame: the goal tolerance wedge, the rod and the applied torque."""
    from PIL import Image, ImageDraw

    size, length = 400, 150
    center = np.array([size / 2, size / 2])
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    def tip(angle: float) -> tuple[float, float]:
        # theta = 0 is upright; image y points down
        return tuple(center + length * np.array([np.sin(angle), -np.cos(angle)]))

    # PIL angles run clockwise from 3 o'clock, so upright (theta = 0) is -90 degrees
    tolerance = np.degrees(2 * np.arcsin(goal_radius / 2))
    heading = np.degrees(goal) - 90
    box = [*(center - length - 20), *(center + length + 20)]
    draw.pieslice(box, heading - tolerance, heading + tolerance, fill=(255, 205, 205))
    draw.ellipse([*(np.array(tip(goal)) - 8), *(np.array(tip(goal)) + 8)], outline="red", width=3)
    draw.line([tuple(center), tip(theta)], fill="black", width=8)
    bob = np.array(tip(theta))
    draw.ellipse([*(bob - 14), *(bob + 14)], fill="green" if success else "royalblue")
    draw.ellipse([*(center - 6), *(center + 6)], fill="black")
    draw.text((10, 10), f"step {step}   torque {torque:+.2f}", fill="black")
    return np.asarray(image)


def rollout_pendulum(
    args: argparse.Namespace, run: Path, config: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    """Roll out a pendulum checkpoint for one episode and save metrics and a real-time animation.

    Args:
        args: Parsed command-line arguments of `main`.
        run: Run directory containing `model.pkl`.
        config: Saved run config with "environment" and "algorithm" sections.
        checkpoint: Unpickled `model.pkl`; its "policy" entry holds the policy params.
    """
    import imageio.v3 as iio
    from ml_collections import config_dict

    from chronosrl.environments.toy_examples.pendulum_env import PendulumEnv

    env_config = config_dict.ConfigDict(config["environment"])
    env = PendulumEnv(env_config, nr_envs=1)
    algorithm = config["algorithm"]
    chunk_length = algorithm["action_chunk_length"]
    policy = Policy(env.action_size * chunk_length, algorithm["width"], algorithm["depth"])
    params = checkpoint["policy"]
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    # Deterministic policy: tanh of the mean, a flat chunk of `chunk_length` actions.
    act = jax.jit(lambda obs: jnp.tanh(policy.apply(params, env.policy_observation(obs))[0]))
    print(f"Rolling out {run.name} ({config['environment']['name']}), seed {args.seed}", flush=True)
    state = reset(jax.random.split(jax.random.PRNGKey(args.seed), 1))
    frames = []
    successes, errors = [], []
    ended = False
    for index in range(args.steps):
        if index % chunk_length == 0:
            chunk = act(state.obs).reshape(1, chunk_length, env.action_size)
        action = chunk[:, index % chunk_length]
        frames.append(
            pendulum_frame(
                float(state.theta[0]),
                float(state.goal[0]),
                env.goal_radius,
                float(action[0, 0]) * env_config.max_torque,
                index,
                bool(state.metrics["success"][0]),
            )
        )
        state = step(state, action)
        successes.append(float(state.metrics["success"][0]))
        errors.append(float(state.metrics["goal_error"][0]))
        if bool(state.done[0]):
            # Same-step autoreset returns a different episode: stop before drawing it.
            ended = True
            break
    output = args.output or run / f"rollout{epoch_suffix(args)}_seed{args.seed}.gif"
    if output.suffix not in (".gif", ".mp4"):
        raise ValueError("Pendulum --output must be a .gif or .mp4 animation")
    output.parent.mkdir(parents=True, exist_ok=True)
    # One frame per policy step, played back in real time
    if output.suffix == ".gif":
        iio.imwrite(output, np.stack(frames), duration=1000 * env_config.dt, loop=0)
    else:
        iio.imwrite(output, np.stack(frames), fps=1 / env_config.dt)
    metrics = {
        "run": str(run),
        "environment": config["environment"]["name"],
        "seed": args.seed,
        "steps": len(successes),
        "time_at_goal": float(np.sum(successes)),
        "goal_reached": bool(np.any(successes)),
        "goal_error": float(np.mean(errors)),
        "episode_ended": ended,
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Animation: {output.resolve()}", flush=True)


def epoch_suffix(args: argparse.Namespace) -> str:
    """Return `_epoch<N>` when a specific epoch is rolled out, so outputs don't overwrite."""
    return "" if args.epoch is None else f"_epoch{args.epoch:03d}"


def main():
    """Parse arguments, load the run's config and checkpoint, and roll out the policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/srl/default"))
    parser.add_argument("--environment", help="Override the saved environment name")
    parser.add_argument(
        "--epoch", type=int, help="Load checkpoints/epoch_<N>.pkl instead of the latest model.pkl"
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=60, help="Playback frames per second")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default="cpu", help="Crazyflow rollout device"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Crazyflow: compute metrics without opening a window",
    )
    parser.add_argument(
        "--loop", action="store_true", help="Crazyflow: repeat playback until the window is closed"
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "survival-reinforcement-learning",
        help="Original SRL repository (needed for args.pkl checkpoints)",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    run = args.run.resolve()
    # A parent directory is also accepted: use its last run (by name) with a model.pkl.
    if not (run / "model.pkl").exists():
        candidates = sorted(run.glob("*/model.pkl"), key=lambda p: p.parent.name)
        if not candidates:
            parser.error(f"No model.pkl found in {run}")
        run = candidates[-1].parent
    config_path = run / "config.json"
    original_config = None
    # Config sources: original SRL args.pkl (if no config.json), the runner's config.json,
    # or as a last resort the matching wandb config.yaml.
    if (run / "args.pkl").exists() and not config_path.exists():
        try:
            from omegaconf import OmegaConf
        except ImportError:
            parser.error("Reading args.pkl requires omegaconf in the benchmark environment")
        with (run / "args.pkl").open("rb") as file:
            original_config = OmegaConf.to_container(pickle.load(file), resolve=True)
        architecture = original_config["architecture"]
        config = {
            "environment": {"name": original_config["env"]["env_id"]},
            "algorithm": {
                "width": architecture["actor_network_width"],
                "depth": architecture["actor_network_depth"],
                "action_chunk_length": 1,
            },
        }
    elif config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        configs = sorted(Path("wandb").glob("run-*-*/files/config.yaml"))
        config = None
        for path in configs:
            saved = yaml.safe_load(path.read_text())
            runner = saved.get("runner", {}).get("value", {})
            if str(runner.get("run_name")) == run.name:
                config = {
                    key: saved[key]["value"] for key in ("algorithm", "environment", "runner")
                }
                break
        if config is None:
            parser.error(
                f"No saved config for {run}; expected config.json or matching wandb config"
            )
    name = args.environment or config["environment"]["name"]
    if name.startswith("go2"):
        parser.error("This viewer currently supports JaxGCRL environments only")
    algorithm = config["algorithm"]
    chunk_length = algorithm["action_chunk_length"]
    checkpoint_path = run / "model.pkl"
    if args.epoch is not None:
        checkpoint_path = run / "checkpoints" / f"epoch_{args.epoch:03d}.pkl"
        if not checkpoint_path.exists():
            parser.error(f"No checkpoint for epoch {args.epoch}: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if name.startswith("crazyflow_"):
        from chronosrl.environments.crazyflow import ENVS

        if name not in ENVS or original_config is not None:
            parser.error(f"Crazyflow playback requires a native checkpoint of one of {list(ENVS)}")
        rollout_crazyflow(args, run, config, checkpoint)
        return
    if name.startswith("pendulum_"):
        if original_config is not None:
            parser.error("Pendulum playback requires a native pendulum checkpoint")
        rollout_pendulum(args, run, config, checkpoint)
        return
    from brax.io import html

    # Apply the project's Brax/MuJoCo compatibility setup for benchmark checkpoints.
    environment_module = get_environment(name)
    if original_config is not None:
        source_repo = args.source_repo.resolve()
        if not (source_repo / "srl/utils/actors.py").exists():
            parser.error("Specify --source-repo pointing to the original SRL repository")
        sys.path.insert(0, str(source_repo))
        from flax import linen as nn
        from srl.envs import make_env
        from srl.utils.actors import Actor

        activation_name = architecture["activation"].lower()
        if activation_name not in ("swish", "silu", "relu", "gelu", "elu"):
            parser.error(f"Unsupported original actor activation: {activation_name}")
        env, _, _, _ = make_env(name)
        policy = Actor(
            env.action_size,
            algorithm["width"],
            algorithm["depth"],
            bool(architecture["use_residuals"]),
            getattr(nn, activation_name),
        )
        if not isinstance(checkpoint, (tuple, list)) or len(checkpoint) != 3:
            parser.error("Original SRL checkpoint must contain (alpha, actor, critic)")
        params = checkpoint[1]

        def policy_observation(obs: jax.Array) -> jax.Array:
            return obs

    else:
        env = environment_module.TASKS[name]()
        policy = Policy(env.action_size * chunk_length, algorithm["width"], algorithm["depth"])
        params = checkpoint["policy"]
        policy_observation = env.policy_observation
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(lambda obs: jnp.tanh(policy.apply(params, policy_observation(obs))[0]))
    print(f"Rolling out {run.name} ({name}), seed {args.seed}", flush=True)
    state = reset(jax.random.PRNGKey(args.seed))
    states = [jax.device_get(state.pipeline_state)]
    successes = []
    for index in range(args.steps):
        if index % chunk_length == 0:
            chunk = act(state.obs).reshape(chunk_length, env.action_size)
        state = step(state, chunk[index % chunk_length])
        states.append(jax.device_get(state.pipeline_state))
        successes.append(float(state.metrics["success"]))
        if bool(state.done):
            break
    output = args.output or run / f"rollout{epoch_suffix(args)}_seed{args.seed}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    # The viewer uses system.dt as the interval between saved rollout frames.
    viewer_sys = jax.device_get(env.sys).replace(dt=1.0 / args.fps)
    output.write_text(html.render(viewer_sys, states, height="100vh", colab=False))
    metrics = {
        "run": str(run),
        "environment": name,
        "seed": args.seed,
        "steps": len(successes),
        "time_at_goal": float(np.sum(successes)),
        "goal_reached": bool(np.any(successes)),
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Viewer: {output.resolve()}")


if __name__ == "__main__":
    main()
