"""Flax networks: residual MLP encoders, survival and contrastive critics, and the policy."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal, variance_scaling, zeros


def dense(features: int) -> nn.Dense:
    """Dense layer with fan-in uniform initialisation and zero bias."""
    return nn.Dense(
        features, kernel_init=variance_scaling(1 / 3, "fan_in", "uniform"), bias_init=zeros
    )


def dense_norm_act(x: jax.Array, width: int) -> jax.Array:
    """Dense -> LayerNorm -> swish."""
    return nn.swish(nn.LayerNorm()(dense(width)(x)))


class ResidualBlock(nn.Module):
    """Four dense-norm-swish layers with a skip connection."""

    width: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the block; the input must already have `width` features."""
        y = x
        for _ in range(4):
            y = dense_norm_act(y, self.width)
        return x + y


class MLP(nn.Module):
    """Plain MLP if `depth < 4`, else an input layer followed by `depth // 4` residual blocks."""

    width: int
    depth: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Maps (..., in_dim) inputs to (..., width) features."""
        if self.depth < 4:
            for _ in range(self.depth):
                x = dense_norm_act(x, self.width)
            return x
        x = dense_norm_act(x, self.width)
        for _ in range(self.depth // 4):
            x = ResidualBlock(self.width)(x)
        return x


class Encoder(nn.Module):
    """MLP followed by a linear projection to an embedding, optionally layer-normalised."""

    width: int
    depth: int
    embedding_dim: int
    layer_norm: bool
    constant_input: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Maps (..., in_dim) inputs to (..., embedding_dim) embeddings."""
        if self.constant_input:
            # Append a constant 1 feature to the input
            x = jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), x.dtype)], axis=-1)
        z = dense(self.embedding_dim)(MLP(self.width, self.depth)(x))
        return nn.LayerNorm()(z) if self.layer_norm else z


class HazardHead(nn.Module):
    """Predicts discrete-time goal hitting-time logits from state-action and goal embeddings."""

    nr_bins: int
    width: int
    depth: int
    embedding_dim: int
    nr_occupancy_bins: int = 0
    rank: int = 64

    @nn.compact
    def __call__(self, z_sa: jax.Array, z_g: jax.Array) -> tuple[jax.Array, ...]:
        """Computes the hazard logits (and the occupancy logits if enabled).

        Args:
            z_sa: State-action embeddings, shape (..., embedding_dim).
            z_g: Goal embeddings, shape (..., embedding_dim).

        Returns:
            `(logit0, logits)`, plus `occupancy_logits` if `nr_occupancy_bins > 0`. `logit0`
            (...,) is the logit of the goal being reached at t = 0, `logits` (..., nr_bins) the
            per-step hazard logits and `occupancy_logits` (..., nr_occupancy_bins) the
            goal-occupancy logits per time bin.
        """
        # FiLM: goal-dependent scale and shift of z_sa, kept close to the identity
        film = dense(2 * self.embedding_dim)(z_g)
        z = (1.0 + 0.1 * film[..., : self.embedding_dim]) * z_sa + 0.1 * film[
            ..., self.embedding_dim :
        ]
        h = dense_norm_act(jnp.concatenate([z, z_g, z - z_g, z * z_g], axis=-1), self.width)
        for _ in range(self.depth // 4):
            h = ResidualBlock(self.width)(h)
        logit0 = dense(1)(h)[..., 0]
        # Low-rank hazard logits: (..., rank) features times a learned (nr_bins, rank) time basis
        basis = self.param("time_basis", normal(0.02), (self.nr_bins, self.rank))
        bias = self.param("time_bias", zeros, (self.nr_bins,))
        logits = dense(self.rank)(h) @ basis.T + bias
        if self.nr_occupancy_bins == 0:
            return logit0, logits
        return logit0, logits, dense(self.nr_occupancy_bins)(h)


class SurvivalCritic(nn.Module):
    """State-action and goal encoders followed by a `HazardHead`."""

    nr_bins: int
    width: int
    depth: int
    encoder_depth: int
    embedding_dim: int
    layer_norm: bool
    goal_constant_input: bool
    nr_occupancy_bins: int = 0

    @nn.compact
    def __call__(
        self, state: jax.Array, action: jax.Array, goal: jax.Array
    ) -> tuple[jax.Array, ...]:
        """Returns the `HazardHead` outputs followed by the embeddings `z_sa` and `z_g`."""
        z_sa = Encoder(self.width, self.encoder_depth, self.embedding_dim, self.layer_norm)(
            jnp.concatenate([state, action], axis=-1)
        )
        z_g = Encoder(
            self.width,
            self.encoder_depth,
            self.embedding_dim,
            self.layer_norm,
            self.goal_constant_input,
        )(goal)
        return HazardHead(
            self.nr_bins, self.width, self.depth, self.embedding_dim, self.nr_occupancy_bins
        )(z_sa, z_g) + (z_sa, z_g)


class ContrastiveCritic(nn.Module):
    """Two-tower critic embedding state-action pairs and goals into a shared space."""

    width: int
    depth: int
    embedding_dim: int
    layer_norm: bool
    goal_constant_input: bool

    @nn.compact
    def __call__(
        self, state: jax.Array, action: jax.Array, goal: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Returns the state-action and goal embeddings `(z_sa, z_g)`."""
        z_sa = Encoder(self.width, self.depth, self.embedding_dim, self.layer_norm)(
            jnp.concatenate([state, action], axis=-1)
        )
        z_g = Encoder(
            self.width, self.depth, self.embedding_dim, self.layer_norm, self.goal_constant_input
        )(goal)
        return z_sa, z_g


class Policy(nn.Module):
    """Gaussian policy; the tanh squashing is applied by the caller."""

    action_size: int
    width: int
    depth: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns the pre-tanh mean and the log std, bounded to [-5, 2]."""
        x = MLP(self.width, self.depth)(x)
        mean = dense(self.action_size)(x)
        log_std = -5.0 + 3.5 * (nn.tanh(dense(self.action_size)(x)) + 1.0)
        return mean, log_std


def pairwise_distances(z_sa: jax.Array, z_g: jax.Array) -> jax.Array:
    """L2 distances between all rows of `z_sa` (B, D) and `z_g` (B', D), shape (B, B')."""
    # The epsilon keeps the gradient of sqrt finite at zero distance
    return jnp.sqrt(jnp.sum((z_sa[:, None, :] - z_g[None, :, :]) ** 2, axis=-1) + 1e-12)
