"""Inspect how a crazyflow_race checkpoint flies at the first gate, from platform starts only.

Rolls out one episode per world with the mean policy (eval goal radius) and with the sampling
policy that collects the training data (training goal radius), and reports:

- whether the first gate's centre waypoint is reached,
- where the drone crosses the gate plane: through the opening, above, below or beside it,
- how episodes end, which separates frame hits from flying out of bounds,
- how much the mean action changes when only the gate input or only the first goal is moved.

python -m chronosrl.evaluation.crazyflow.drone_race_inspect --run <run dir>
python -m chronosrl.evaluation.crazyflow.drone_race_inspect --run <run dir> \
    --checkpoint <run dir>/checkpoints/epoch_010.pkl
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
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import config_dict

HALF_OPENING = 0.2  # m, gate.xml: the collision boxes start here
HALF_OUTER = 0.36  # m, gate.xml: the collision boxes end here
FAR = 1.0  # m, crossings further than this from the centre are not attempts at the gate
GATE_STATE_START = 18  # the gate features follow the 18 drone state entries in the observation
PERTURBATION = 0.15  # m, the gate position noise of the track


def rollout(
    env: Any, policy: Any, params: Any, nr_envs: int, steps: int, key: jax.Array, stochastic: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Roll out ``steps`` policy steps; returns the ``(T, n, ...)`` trajectories and the task."""

    def act(obs: jax.Array, action_key: jax.Array) -> jax.Array:
        mean, log_std = policy.apply(params, env.policy_observation(obs))
        if stochastic:
            mean = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
        return jnp.tanh(mean)

    def step(carry: tuple[Any, jax.Array], _: None) -> tuple[tuple[Any, jax.Array], dict]:
        state, key = carry
        key, action_key = jax.random.split(key)
        next_state = env.step(state, act(state.obs, action_key))
        output = {
            "obs": state.obs,
            "pos": state.data.states.pos[:, 0],
            "vel": state.data.states.vel[:, 0],
            "done": next_state.done,
            "truncation": next_state.info["truncation"],
            "reached": next_state.metrics["waypoint_reached"],  # measured before the autoreset
        }
        return (next_state, key), output

    reset_key, scan_key = jax.random.split(key)
    state = jax.jit(env.reset)(jax.random.split(reset_key, nr_envs))
    scan = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=steps)[1])
    trajectory = jax.tree.map(np.asarray, scan((state, scan_key)))
    return trajectory, jax.tree.map(np.asarray, state.task)


def to_gate_frames(points: np.ndarray, task: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``points`` of shape ``(..., 3)`` per world in every gate's frame: ``(..., n, G, 3)``.

    ``points`` must have the world axis second to last: ``(n, 3)`` or ``(T, n, 3)``.
    """
    n, G = task["gates_pos"].shape[:2]
    # Columns are the gate axes in world coordinates: x is the normal, the passing direction
    rot = np.asarray(R.from_quat(jnp.asarray(task["gates_quat"]).reshape(-1, 4)).as_matrix())
    rot = rot.reshape(n, G, 3, 3)
    relative = points[..., None, :] - task["gates_pos"]
    return np.einsum("ngji,...ngj->...ngi", rot, relative)


def crossing_regions(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Name the part of the gate plane each ``(y, z)`` crossing point lies in."""
    return np.select(
        [
            (np.abs(y) < HALF_OPENING) & (np.abs(z) < HALF_OPENING),
            np.maximum(np.abs(y), np.abs(z)) > FAR,
            (np.abs(z) >= np.abs(y)) & (z > 0),
            np.abs(z) >= np.abs(y),
        ],
        ["through opening", "far from gate", "above", "below"],
        default="beside",
    )


def analyse(
    trajectory: dict[str, np.ndarray],
    task: dict[str, np.ndarray],
    gate: int,
    centre_waypoint: int,
    bounds: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Summarize the first episode of every world at gate index ``gate``.

    Returns:
        The summary and the ``(T, n)`` mask of steps inside each world's first episode.
    """
    done = trajectory["done"] > 0
    pos, vel = trajectory["pos"], trajectory["vel"]
    T, n = done.shape
    worlds = np.arange(n)
    alive = np.concatenate([np.ones((1, n), bool), np.cumsum(done, axis=0)[:-1] == 0])
    last = np.sum(alive, axis=0) - 1

    # The autoreset hides the final position, so extrapolate one step from the last one
    end = pos[last, worlds] + dt * vel[last, worlds]
    end_local = to_gate_frames(end, task)  # (n, G, 3)
    near = (np.abs(end_local[..., 0]) < 0.2) & (
        np.maximum(np.abs(end_local[..., 1]), np.abs(end_local[..., 2])) < HALF_OUTER + 0.09
    )
    stand = (
        (np.abs(end_local[..., 0]) < 0.2)
        & (np.abs(end_local[..., 1]) < 0.2)
        & (end_local[..., 2] <= -HALF_OUTER)
    )
    others = np.delete(np.arange(task["gates_pos"].shape[1]), gate)
    survived = done[last, worlds] & (trajectory["truncation"][last, worlds] > 0)
    survived |= ~done[last, worlds]
    low, high = bounds
    ends = np.select(
        [
            survived,
            end[:, 2] <= low[2],
            np.any((end < low) | (end > high), axis=-1),
            near[worlds, gate],
            stand[worlds, gate],
            np.any(near[:, others] | stand[:, others], axis=-1),
        ],
        ["survived", "floor", "out of bounds", "gate frame", "gate stand", "other gate"],
        default="other contact",
    )

    local = to_gate_frames(pos, task)[:, worlds, gate]  # (T, n, 3)
    x = local[..., 0]
    forward = alive[1:] & (x[:-1] < 0.0) & (x[1:] >= 0.0)
    crossed = np.any(forward, axis=0)
    first = np.argmax(forward, axis=0)  # the step before the crossing
    fraction = -x[first, worlds] / np.maximum(x[first + 1, worlds] - x[first, worlds], 1e-6)
    at_plane = local[first, worlds] + fraction[:, None] * (
        local[first + 1, worlds] - local[first, worlds]
    )
    regions = np.where(crossed, crossing_regions(at_plane[:, 1], at_plane[:, 2]), "never crossed")

    distance = np.where(alive, np.linalg.norm(local, axis=-1), np.inf)
    speed = np.linalg.norm(vel[first, worlds], axis=-1)
    above = regions == "above"
    summary = {
        "episodes": n,
        "mean_length": float(np.mean(last + 1)),
        "reached_gate_centre": float(
            np.mean(np.any(trajectory["reached"][..., centre_waypoint] * alive > 0, axis=0))
        ),
        "closest_to_centre_m": {
            "median": float(np.median(np.min(distance, axis=0))),
            "p10": float(np.percentile(np.min(distance, axis=0), 10)),
        },
        "ends": {name: float(np.mean(ends == name)) for name in np.unique(ends)},
        "crossings": {name: float(np.mean(regions == name)) for name in np.unique(regions)},
        "median_speed_at_crossing_m_s": float(np.median(speed[crossed])) if crossed.any() else None,
        "median_height_above_centre_when_above_m": float(np.median(at_plane[above, 2]))
        if above.any()
        else None,
        "cleared_frame_when_above": float(np.mean(at_plane[above, 2] > HALF_OUTER))
        if above.any()
        else None,
    }
    # Steps still on the way to the gate, used for the input sensitivity
    approaching = (
        alive
        & (np.where(crossed, np.arange(T)[:, None] <= first, True))
        & ((distance > 0.3) & (distance < 1.2))
    )
    return summary, approaching


def sensitivity(
    env: Any, policy: Any, params: Any, obs: jax.Array, gate_start: int
) -> dict[str, float]:
    """Mean change of the mean action when only the gate or only the first goal input moves."""

    def mean_action(observation: jax.Array) -> jax.Array:
        return jnp.tanh(policy.apply(params, env.policy_observation(observation))[0])

    base = mean_action(obs)
    offsets = PERTURBATION * jnp.concatenate([jnp.eye(3), -jnp.eye(3)])

    def change(start: int) -> float:
        def one(offset: jax.Array) -> jax.Array:
            moved = obs.at[:, start : start + 3].add(offset)
            return jnp.linalg.norm(mean_action(moved) - base, axis=-1)

        return float(jnp.mean(jax.vmap(one)(offsets)))

    log_std = policy.apply(params, env.policy_observation(obs))[1]
    return {
        "gate_input_moved": change(gate_start),
        "first_goal_moved": change(env.state_dim),
        "policy_std_for_scale": float(jnp.mean(jnp.exp(log_std))),
        "states": int(obs.shape[0]),
    }


def main() -> None:
    """Load the checkpoint, roll out both policies and print the summaries as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Run directory with config.json")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to <run>/model.pkl")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    # The default leaves the GPU to running training jobs
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()
    jax.config.update("jax_platforms", args.device)

    # Imported here so the platform choice applies to the simulator as well
    from chronosrl.environments.crazyflow import eval_config, make_env
    from chronosrl.environments.crazyflow.wrappers import Wrapper
    from chronosrl.networks import Policy

    saved = json.loads((args.run / "config.json").read_text())
    with open(args.checkpoint or args.run / "model.pkl", "rb") as file:
        params = pickle.load(file)["policy"]

    algorithm = saved["algorithm"]
    assert algorithm["action_chunk_length"] == 1, "only single-step actions are supported"
    config = config_dict.ConfigDict(saved["environment"])
    config.device = args.device
    assert config.name == "crazyflow_race", "this script inspects crazyflow_race runs"

    sampling = config.copy_and_resolve_references()
    sampling.task.spawn_at_gates = False
    bounds = np.array(config.bounds.low), np.array(config.bounds.high)
    key = jax.random.PRNGKey(args.seed)
    results: dict[str, Any] = {}
    for name, env_config, stochastic in (
        ("mean policy, eval radius", eval_config(config), False),
        ("sampling policy, training radius", sampling, True),
    ):
        key, rollout_key = jax.random.split(key)
        env = make_env(env_config, args.episodes)
        race = env
        while isinstance(race, Wrapper):
            race = race.env
        policy = Policy(env.action_size, algorithm["width"], algorithm["depth"])
        trajectory, task = rollout(
            env, policy, params, args.episodes, config.episode_length, rollout_key, stochastic
        )
        gate = int(race.gate_sequence[0])
        centre_waypoint = 0 if race.centres_only else 1
        summary, approaching = analyse(
            trajectory, task, gate, centre_waypoint, bounds, 1.0 / config.freq
        )
        results[name] = summary
        if not stochastic:
            states = trajectory["obs"][approaching]
            picked = np.random.default_rng(args.seed).permutation(len(states))[:2000]
            results["action change for a 0.15 m input shift"] = (
                sensitivity(
                    env, policy, params, jnp.asarray(states[picked]), GATE_STATE_START + 6 * gate
                )
                if len(picked)
                else "no approaching states"
            )
        env.sim.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
