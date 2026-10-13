"""Roll out a saved JaxGCRL policy and export an interactive Brax viewer."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax.io import html

from chronosrl.environments import get_environment
from chronosrl.networks import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/srl/default"))
    parser.add_argument("--environment", help="Override the saved environment name")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=60, help="Playback frames per second")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-repo", type=Path,
                        default=Path(__file__).resolve().parents[2] / "survival-reinforcement-learning",
                        help="Original SRL repository (needed for args.pkl checkpoints)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    run = args.run.resolve()
    if not (run / "model.pkl").exists():
        candidates = sorted(run.glob("*/model.pkl"), key=lambda p: p.parent.name)
        if not candidates:
            parser.error(f"No model.pkl found in {run}")
        run = candidates[-1].parent
    config_path = run / "config.json"
    original_config = None
    if (run / "args.pkl").exists() and not config_path.exists():
        try:
            from omegaconf import OmegaConf
        except ImportError:
            parser.error("Reading args.pkl requires omegaconf in the benchmark environment")
        with (run / "args.pkl").open("rb") as file:
            original_config = OmegaConf.to_container(pickle.load(file), resolve=True)
        architecture = original_config["architecture"]
        config = {"environment": {"name": original_config["env"]["env_id"]},
                  "algorithm": {"width": architecture["actor_network_width"],
                                "depth": architecture["actor_network_depth"],
                                "action_chunk_length": 1}}
    elif config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        configs = sorted(Path("wandb").glob(f"run-*-*/files/config.yaml"))
        config = None
        for path in configs:
            saved = yaml.safe_load(path.read_text())
            runner = saved.get("runner", {}).get("value", {})
            if str(runner.get("run_name")) == run.name:
                config = {key: saved[key]["value"] for key in ("algorithm", "environment", "runner")}
                break
        if config is None:
            parser.error(f"No saved config for {run}; expected config.json or matching wandb config")
    name = args.environment or config["environment"]["name"]
    if name.startswith("go2"):
        parser.error("This viewer currently supports JaxGCRL environments only")
    # Apply the project's Brax/MuJoCo compatibility setup for both checkpoint formats.
    environment_module = get_environment(name)
    algorithm = config["algorithm"]
    chunk_length = algorithm["action_chunk_length"]
    with (run / "model.pkl").open("rb") as file:
        checkpoint = pickle.load(file)
    if original_config is not None:
        source_repo = args.source_repo.resolve()
        if not (source_repo / "srl/utils/actors.py").exists():
            parser.error("Specify --source-repo pointing to the original SRL repository")
        sys.path.insert(0, str(source_repo))
        from srl.envs import make_env
        from srl.utils.actors import Actor
        from flax import linen as nn
        activation_name = architecture["activation"].lower()
        if activation_name not in ("swish", "silu", "relu", "gelu", "elu"):
            parser.error(f"Unsupported original actor activation: {activation_name}")
        env, _, _, _ = make_env(name)
        policy = Actor(env.action_size, algorithm["width"], algorithm["depth"],
                       bool(architecture["use_residuals"]), getattr(nn, activation_name))
        if not isinstance(checkpoint, (tuple, list)) or len(checkpoint) != 3:
            parser.error("Original SRL checkpoint must contain (alpha, actor, critic)")
        params = checkpoint[1]
        policy_observation = lambda obs: obs
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
    output = args.output or run / f"rollout_seed{args.seed}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    # The viewer uses system.dt as the interval between saved rollout frames.
    viewer_sys = jax.device_get(env.sys).replace(dt=1.0 / args.fps)
    output.write_text(html.render(viewer_sys, states, height="100vh", colab=False))
    metrics = {"run": str(run), "environment": name, "seed": args.seed,
               "steps": len(successes), "time_at_goal": float(np.sum(successes)),
               "goal_reached": bool(np.any(successes))}
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Viewer: {output.resolve()}")


if __name__ == "__main__":
    main()
