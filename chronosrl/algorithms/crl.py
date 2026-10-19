"""CRL: contrastive RL with a negative-L2-distance critic trained with InfoNCE."""

from typing import Any

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.networks import ContrastiveCritic, pairwise_distances


def get_config(algorithm_name: str) -> config_dict.ConfigDict:
    """Default CRL hyperparameters."""
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.total_timesteps = 100_000_000
    config.nr_epochs = 100
    config.unroll_length = 62
    config.min_replay_size = 1000
    config.max_replay_size = 10000
    config.nr_windows = -1
    config.batch_size = 512
    config.nr_sgd_batches = 800
    config.learning_rate = 3e-4
    config.width = 256
    config.depth = 8
    config.embedding_dim = 64
    config.embedding_layer_norm = False
    config.window = 1000
    config.goal_probabilities = (1.0, 0.0, 0.0)
    config.future_goal_gamma = 0.99
    config.goal_sequence_length = 1
    config.action_chunk_length = 1
    config.goal_constant_input = False
    config.entropy_per_action_dim = -0.5
    config.logsumexp_penalty = 0.1

    return config


class CRL:
    """Contrastive RL: the critic is the negative L2 distance between the embeddings."""

    def __init__(self, config: config_dict.ConfigDict, env: Any) -> None:
        """Builds the contrastive critic."""
        self.state_dim = env.state_dim
        self.critic = ContrastiveCritic(
            config.width,
            config.depth,
            config.embedding_dim,
            config.embedding_layer_norm,
            config.goal_constant_input,
        )
        self.logsumexp_penalty = config.logsumexp_penalty
        self.occupancy_edges = None

    def critic_loss(
        self, params: Any, batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """InfoNCE with in-batch negatives plus a logsumexp penalty; returns loss and metrics."""
        z_sa, z_g = self.critic.apply(
            params, batch["observation"][:, : self.state_dim], batch["action"], batch["goal"]
        )
        # Logits are -distances[i, j]: row i's positive is its own goal (the diagonal),
        # the other goals of the batch are its negatives
        distances = pairwise_distances(z_sa, z_g)
        logsumexp = jax.nn.logsumexp(-distances, axis=1)
        # Only samples whose goal is reached within the window count as positives
        positive = batch["is_event"] * batch["valid"]
        infonce = jnp.sum(positive * (jnp.diagonal(distances) + logsumexp)) / jnp.maximum(
            jnp.sum(positive), 1.0
        )
        loss = infonce + self.logsumexp_penalty * jnp.mean(logsumexp**2)
        return loss, {"loss/infonce_loss": infonce, "loss/critic_loss": loss}

    def actor_value(
        self,
        params: Any,
        state: jax.Array,
        action: jax.Array,
        goal: jax.Array,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        """Negative L2 distance between the state-action and goal embeddings (`batch` unused)."""
        z_sa, z_g = self.critic.apply(params, state, action, goal)
        return -jnp.sqrt(jnp.sum((z_sa - z_g) ** 2, axis=-1) + 1e-12)
