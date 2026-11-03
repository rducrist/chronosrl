"""Map which maze goals a crazyflow_maze policy reaches from the start cell, per epoch checkpoint.

Every free cell gets ``k x k`` goals, one world each. All drones start at rest in the start cell,
without randomization, and the policy acts with its mean action, so one episode per goal and
checkpoint is deterministic. A goal counts as reached if the drone enters ``goal_radius`` of it
before the episode ends. Running this over the ``checkpoints/epoch_<N>.pkl`` of a run shows how the
reachable region grows during training.

python -m chronosrl.evaluation.crazyflow.drone_maze_reachability --run runs/crazyflow_maze/srl
python -m chronosrl.evaluation.crazyflow.drone_maze_reachability --run <run> --plot-only
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from chronosrl.environments.maze_layouts import LAYOUTS
from chronosrl.environments.maze_layouts import R as RESET


def deterministic_config(saved: dict, device: str) -> config_dict.ConfigDict:
    """Return the run's environment config with every source of randomness switched off."""
    env_config = config_dict.ConfigDict(saved)
    env_config.device = device
    env_config.randomization.enabled = False
    env_config.maze.random_start = False
    env_config.maze.spawn_noise = 0.0
    env_config.spawn.max_speed = 0.0
    env_config.spawn.low = env_config.spawn.high
    return env_config


def goal_grid(
    layout: list[list], scaling: float, k: int, z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n, 3)`` goals at the centres of a ``k x k`` split of every free cell.

    Also returns their ``(n, 2)`` pixel indices ``(i * k + a, j * k + b)`` in the
    ``(H * k, W * k)`` reachability image, where cell ``(i, j)`` is centred at ``(i, j) * scaling``.
    """
    offsets = ((np.arange(k) + 0.5) / k - 0.5) * scaling  # sub-cell centres around the cell centre
    goals, pixels = [], []
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == 1:
                continue
            for a, dx in enumerate(offsets):
                for b, dy in enumerate(offsets):
                    goals.append([i * scaling + dx, j * scaling + dy, z])
                    pixels.append([i * k + a, j * k + b])
    return np.array(goals, np.float32), np.array(pixels)


def evaluate(
    env_config: config_dict.ConfigDict, algorithm: dict, checkpoints: list[Path], goals: np.ndarray
) -> np.ndarray:
    """Return the ``(n_checkpoints, n_goals)`` reached flags of one episode per goal."""
    # Imported here so --plot-only works without the simulator
    from chronosrl.environments.crazyflow import EvalWrapper, make_env
    from chronosrl.environments.crazyflow.drone_env import State
    from chronosrl.networks import Policy

    env = EvalWrapper(make_env(env_config, len(goals)))
    policy = Policy(env.action_size, algorithm["width"], algorithm["depth"])
    goals = jnp.asarray(goals)

    # params is an argument, not a closure, so all checkpoints share one compilation
    @jax.jit
    def run(params: Any) -> jax.Array:
        def body(state: State, _: None) -> tuple[State, None]:
            action = jnp.tanh(policy.apply(params, env.policy_observation(state.obs))[0])
            return env.step(state, action), None

        state = env.reset(jax.random.split(jax.random.PRNGKey(0), len(goals)))
        # Replace the sampled goals; the observation holds them in maze units after the state
        state = state.replace(
            goal=goals, obs=state.obs.at[:, env.state_dim :].set(goals / env.scaling)
        )
        state, _ = jax.lax.scan(body, state, None, length=env.episode_length)
        # EvalWrapper only accumulates each world's first episode
        return state.info["episode_metrics"]["success"] > 0

    reached = []
    for path in checkpoints:
        with path.open("rb") as file:
            params = pickle.load(file)["policy"]
        reached.append(np.asarray(run(params)))
        print(f"{path.name}: {reached[-1].mean():.2f} of the goals reached", flush=True)
    env.sim.close()
    return np.stack(reached)


def plot(results: dict[str, np.ndarray], layout: list[list], k: int, epochs: list[int], out: Path):
    """Draw the reachability image of each epoch in ``epochs`` side by side and save it."""
    import matplotlib.pyplot as plt

    shape = (len(layout) * k, len(layout[0]) * k)
    start = next((i, j) for i, row in enumerate(layout) for j, c in enumerate(row) if c == RESET)
    rows = [list(results["epochs"]).index(epoch) for epoch in epochs]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.6")  # walls
    fig, axes = plt.subplots(1, len(rows), figsize=(2.2 * len(rows), 2.6), squeeze=False)
    for ax, row, epoch in zip(axes[0], rows, epochs):
        image = np.full(shape, np.nan)
        pixels = results["pixels"]
        image[pixels[:, 0], pixels[:, 1]] = results["reached"][row]
        # Rows are maze x (cell i), columns maze y (cell j), as the layouts are written
        mappable = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.plot(start[1] * k + (k - 1) / 2, start[0] * k + (k - 1) / 2, "r*", markersize=10)
        ax.set_title(f"epoch {epoch}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(mappable, ax=axes[0].tolist(), shrink=0.8, label="goal reached")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


def main():
    """Evaluate every epoch checkpoint of a run on the goal grid, save the flags and plot them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    parser.add_argument("--k", type=int, default=3, help="Goals per cell side")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--epochs", type=int, nargs="+", help="Epochs to plot (default: 8 evenly spaced ones)"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot the saved reachability.npz again"
    )
    args = parser.parse_args()

    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    if config["environment"]["name"] != "crazyflow_maze":
        parser.error(f"{run} is a {config['environment']['name']} run, not crazyflow_maze")
    layout = LAYOUTS[config["environment"]["maze"]["layout"]]
    data_path = run / "reachability.npz"

    if args.plot_only:
        results = dict(np.load(data_path))
        if results["k"] != args.k:
            parser.error(f"{data_path} was computed with --k {results['k']}")
    else:
        if config["algorithm"]["action_chunk_length"] != 1:
            parser.error("Only action_chunk_length = 1 is supported")
        checkpoints = sorted((run / "checkpoints").glob("epoch_*.pkl"))
        if not checkpoints:
            parser.error(f"No checkpoints in {run / 'checkpoints'}")

        env_config = deterministic_config(config["environment"], args.device)
        scaling = env_config.maze.scaling
        goals, pixels = goal_grid(layout, scaling, args.k, env_config.goal.high[2])
        print(f"{len(checkpoints)} checkpoints x {len(goals)} goals", flush=True)
        reached = evaluate(env_config, config["algorithm"], checkpoints, goals)
        epochs = np.array([int(path.stem.split("_")[1]) for path in checkpoints])
        results = {"epochs": epochs, "goals": goals, "pixels": pixels, "reached": reached}
        np.savez(data_path, k=args.k, **results)
        print(f"Saved {data_path}", flush=True)

    epochs = args.epochs
    if epochs is None:
        index = np.unique(np.linspace(0, len(results["epochs"]) - 1, 8).round().astype(int))
        epochs = [int(results["epochs"][i]) for i in index]
    plot(results, layout, args.k, epochs, run / "reachability.pdf")


if __name__ == "__main__":
    main()
