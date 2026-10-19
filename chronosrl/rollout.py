"""Roll out saved policies: Crazyflow native viewer or JaxGCRL HTML playback.

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
    """Roll out a native crazyflow_position checkpoint, save metrics and replay it in MuJoCo.

    Args:
        args: Parsed command-line arguments of `main`.
        run: Run directory containing `model.pkl`.
        config: Saved run config with "environment" and "algorithm" sections.
        checkpoint: Unpickled `model.pkl`; its "policy" entry holds the policy params.
    """
    import mujoco
    from ml_collections import config_dict

    from chronosrl.environments.crazyflow import CrazyflowPositionEnv

    env_config = config_dict.ConfigDict(config["environment"])
    env_config.device = args.device
    env = CrazyflowPositionEnv(env_config, nr_envs=1)
    dt = 1.0 / env_config.freq
    algorithm = config["algorithm"]
    chunk_length = algorithm["action_chunk_length"]
    policy = Policy(env.action_size * chunk_length, algorithm["width"], algorithm["depth"])
    params = checkpoint["policy"]
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    # Deterministic policy: tanh of the mean, a flat chunk of `chunk_length` actions.
    act = jax.jit(lambda obs: jnp.tanh(policy.apply(params, env.policy_observation(obs))[0]))
    print(
        f"Rolling out {run.name} (crazyflow_position), seed {args.seed}, device {args.device}",
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
        "environment": "crazyflow_position",
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
                env.sim.render(world=0)
                time.sleep(max(0.0, started + (index + 1) / args.fps - time.perf_counter()))
            if not args.loop:
                break
    finally:
        env.sim.close()


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
        if name != "crazyflow_position" or original_config is not None:
            parser.error("Crazyflow playback requires a native crazyflow_position checkpoint")
        rollout_crazyflow(args, run, config, checkpoint)
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
