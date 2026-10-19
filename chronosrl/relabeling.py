"""Hindsight relabeling of replay windows into goal hitting-time (survival) training batches."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict


def geometric_bins(horizon: int, nr_bins: int) -> np.ndarray:
    """Time-bin edges from 0 to `horizon`, geometrically spaced (unit bins if `nr_bins >= horizon`).

    Edges that coincide after rounding are merged, so there can be fewer than `nr_bins` bins.
    """
    if nr_bins >= horizon:
        return np.arange(horizon + 1)
    edges = jnp.round(jnp.power(horizon / 1.0, 1.0 / nr_bins) ** jnp.arange(nr_bins + 1)).astype(
        jnp.int32
    )
    return np.unique(np.concatenate([[0, horizon], np.clip(np.asarray(edges), 0, horizon)]))


def bin_discounts(edges: np.ndarray, gamma: float) -> jax.Array:
    """Summed discount `sum_k gamma**k` over each bin `[edges[i], edges[i + 1])`."""
    edges = jnp.asarray(edges)
    starts, lengths = edges[:-1].astype(jnp.float32), (edges[1:] - edges[:-1]).astype(jnp.float32)
    return jnp.power(gamma, starts) * (1.0 - jnp.power(gamma, lengths)) / (1.0 - gamma)


def make_relabel_fn(
    config: config_dict.ConfigDict, env: Any, occupancy_edges: np.ndarray | None = None
) -> Callable[[dict[str, jax.Array], jax.Array], dict[str, jax.Array]]:
    """Builds the function that turns a replay window into a relabeled training batch.

    Args:
        config: Algorithm config (goal sampling, window, action chunk length, cost weight, ...).
        env: Training environment, providing `state_dim`, `goal_indices` and `goal_radius`.
        occupancy_edges: Time-bin edges of the occupancy targets, or None to skip them.

    Returns:
        `relabel(transitions, key)` for a single window; the caller vmaps it over windows.
    """
    state_dim = env.state_dim
    goal_start, goal_end = env.goal_indices
    goal_size = goal_end - goal_start
    radius = env.goal_radius
    m = config.goal_sequence_length
    chunk = config.action_chunk_length
    window = config.window
    cost_weight = config.get("cost_weight", 0.0)
    probabilities = jnp.array(config.goal_probabilities, dtype=jnp.float32)
    probabilities = probabilities / jnp.sum(probabilities)

    def relabel(transitions: dict[str, jax.Array], key: jax.Array) -> dict[str, jax.Array]:
        """Relabels one window of transitions with hindsight goals and hitting times.

        Args:
            transitions: J consecutive transitions of one environment, with keys "observation",
                "action", "discount", "truncation", "seed" (episode id) and "cost".
            key: PRNG key for goal sampling.

        Returns:
            Batch of T = J - (m - 1) samples with keys "observation" (state + first goal),
            "goal" (m goals), "action" (action chunk), "valid", "is_event" (goal reached within
            the window), "tau" (hitting time), "censor" (censoring time), "terminates" and,
            with occupancy edges, "occupancy" / "occupancy_mask" of shape (T, nr_bins).
        """
        # Rows t are the T samples (anchors), columns j the J transitions of the window
        J = transitions["observation"].shape[0]
        T = J - (m - 1)
        j = jnp.arange(J)
        t = jnp.arange(T)
        achieved = transitions["observation"][:, goal_start:goal_end]
        seed = transitions["seed"].astype(jnp.int32)
        same_episode = seed[:T, None] == seed[None, :]
        dt = j[None, :] - t[:, None]

        # Goals are sequences of m consecutive achieved goals from one episode,
        # a single goal for m = 1
        sequence_start = j < T
        for i in range(1, m):
            sequence_start &= jnp.concatenate([seed[:-i] == seed[i:], jnp.zeros((i,), dtype=bool)])

        def sequences(starts: jax.Array) -> jax.Array:
            """Flattened goal sequences starting at `starts`, shape (len(starts), m * goal_size)."""
            return achieved[starts[:, None] + jnp.arange(m)[None, :]].reshape(
                len(starts), m * goal_size
            )

        # Goal strategies: 0 = future goal of the same episode (geometric in dt),
        # 1 = the currently achieved goal, 2 = a random achieved goal from the window
        key_strategy, key_future, key_random = jax.random.split(key, 3)
        future = same_episode & (dt >= 1) & sequence_start[None, :]
        future_index = jax.random.categorical(
            key_future,
            jnp.log(
                jnp.where(future, jnp.power(config.future_goal_gamma, dt.astype(jnp.float32)), 0.0)
                + 1e-10
            ),
            axis=1,
        )
        future_index = jnp.where(jnp.any(future, axis=1), future_index, t)
        random_goal = jnp.tile(achieved[jax.random.randint(key_random, (T,), 0, T)], (1, m))
        strategy = jax.random.choice(key_strategy, 3, shape=(T,), p=probabilities)[:, None]
        goal = jnp.where(
            strategy == 0,
            sequences(future_index),
            jnp.where(strategy == 1, sequences(t), random_goal),
        )
        point_goal = goal[:, :goal_size]

        # Goal-reaching time tau: first step at which the whole goal sequence is matched,
        # 0 if already matched
        # With per-step costs, time runs on the shaped clock
        # C_{t+1} = C_t + 1 + cost_weight * cost_t
        # match[t, s]: the goal of sample t is matched by the achieved sequence starting at s
        candidates = sequences(t).reshape(T, m, goal_size)
        match = jnp.all(
            jnp.linalg.norm(goal.reshape(T, m, goal_size)[:, None] - candidates[None], axis=-1)
            <= radius,
            axis=-1,
        )
        future_match = match & same_episode[:, :T] & (dt[:, :T] >= 1)
        big = jnp.iinfo(jnp.int32).max  # Sentinel for "not reached"
        if cost_weight > 0.0:
            clock = 1.0 + cost_weight * jnp.maximum(transitions["cost"], 0.0)
            shaped_time = jnp.cumsum(clock) - clock
            shaped_dt = jnp.round(shaped_time[None, :T] - shaped_time[:T, None]).astype(jnp.int32)
            first = jnp.min(jnp.where(future_match & (shaped_dt <= window), shaped_dt, big), axis=1)
            censor = jnp.clip(jnp.round(shaped_time[T - 1] - shaped_time[:T]), 0, window).astype(
                jnp.int32
            )
        else:
            first = jnp.min(jnp.where(future_match, dt[:, :T], big), axis=1)
            censor = jnp.full((T,), window, dtype=jnp.int32)
        reached_now = jnp.diagonal(match)
        is_event = reached_now | (first < big)
        tau = jnp.clip(jnp.where(reached_now, 0, first), 0, window)

        # Action chunks are valid only inside the window and the episode
        index = t[:, None] + jnp.arange(chunk)[None, :]
        index_clipped = jnp.clip(index, 0, T - 1)
        chunk_ok = (index < T) & (seed[index_clipped] == seed[:T, None])
        action = jnp.where(chunk_ok[..., None], transitions["action"][index_clipped], 0.0).reshape(
            T, -1
        )

        # Episode ends: the done transition has discount 0,
        # its truncation flag is stored on the next transition
        done = transitions["discount"] < 0.5
        next_truncation = transitions["truncation"][jnp.clip(j + 1, 0, J - 1)]
        termination = done & (next_truncation < 0.5) & (j + 1 < J)
        terminates = jnp.any(
            same_episode & termination[None, :] & (j[None, :] >= t[:, None]), axis=1
        )

        batch = {
            "observation": jnp.concatenate(
                [transitions["observation"][:T, :state_dim], point_goal], axis=-1
            ),
            "goal": goal,
            "action": action,
            "valid": jnp.all(chunk_ok, axis=1).astype(jnp.float32),
            "is_event": is_event.astype(jnp.float32),
            "tau": tau,
            "censor": censor,
            "terminates": terminates.astype(jnp.float32),
        }

        if occupancy_edges is not None:
            # Occupancy of the nested goal regions t steps later, zero after a termination,
            # unobserved after a truncation
            ended = jnp.any(same_episode & done[None, :], axis=1)
            last = jnp.where(
                ended,
                jnp.max(jnp.where(same_episode & done[None, :], j[None, :], -1), axis=1),
                jnp.max(jnp.where(same_episode, j[None, :], -1), axis=1),
            )
            alive_future = same_episode & (j[None, :] <= last[:, None]) & (dt >= 1)
            distance = jnp.linalg.norm(point_goal[:, None, :] - achieved[None, :, :], axis=-1)
            # Soft hit in [0, 1]: fraction of the nested balls (radius * 0.5**i) containing the
            # achieved goal
            inside = (
                sum(
                    (distance <= radius * 0.5**i).astype(jnp.float32)
                    for i in range(config.occupancy_radii)
                )
                / config.occupancy_radii
            )
            truncated = (transitions["truncation"][jnp.clip(last + 1, 0, J - 1)] > 0.5) | (
                last + 1 >= J
            )
            after_termination = (
                (j[None, :] > last[:, None]) & (dt >= 1) & (ended & ~truncated)[:, None]
            )
            hits = alive_future * inside
            observed = (alive_future | after_termination).astype(jnp.float32)
            edges = jnp.asarray(occupancy_edges)
            nr_bins = len(occupancy_edges) - 1
            # Sum hits and observations of each sample over the time offsets in each bin
            if cost_weight > 0.0:
                bins = jnp.searchsorted(
                    edges,
                    jnp.round(shaped_time[None, :] - shaped_time[:T, None])
                    .astype(jnp.int32)
                    .reshape(-1),
                    side="right",
                )
                assign = jax.nn.one_hot(jnp.clip(bins - 1, 0, nr_bins - 1).reshape(T, J), nr_bins)
                hits, observed = (
                    jnp.einsum("tj,tjk->tk", hits, assign),
                    jnp.einsum("tj,tjk->tk", observed, assign),
                )
            else:
                # Re-index columns from the absolute index t + j to the offset j
                offset_index = t[:, None] + j[None, :]
                in_window = (offset_index < J).astype(jnp.float32)
                offset_index = jnp.clip(offset_index, 0, J - 1)
                assign = jax.nn.one_hot(
                    jnp.clip(jnp.searchsorted(edges, j, side="right") - 1, 0, nr_bins - 1), nr_bins
                )
                hits = (jnp.take_along_axis(hits, offset_index, axis=1) * in_window) @ assign
                observed = (
                    jnp.take_along_axis(observed, offset_index, axis=1) * in_window
                ) @ assign
            batch["occupancy"] = hits / jnp.maximum(observed, 1.0)
            batch["occupancy_mask"] = (observed > 0).astype(jnp.float32)

        return batch

    return relabel
