"""Alpha policy network: a small fully-convolutional ResNet (Bois et al. 2024
Fig. 5 architecture) that maps per-element features to the hybrid-DGSEM
blending factor alpha in [0, 1].

Input : (in_channels, n_elem)  — e.g. the normalized modal-energy spectrum
        from jax_dgsem.indicator.modal_energy (in_channels = P+1), transposed.
Output: (n_elem,) raw alpha in (0, 1) via sigmoid.

Fully convolutional over the element axis -> the same weights run on any
number of elements (train on coarse meshes, deploy on finer ones).

Pure architecture: no DG knowledge, no training logic (training lives in
nn/training/train.py; eqx modules are immutable pytrees, so an optimizer
cannot be a mutable attribute here anyway).
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class ResBlock(eqx.Module):
    """Two same-width convolutions with a skip connection (Fig. 5 inner block)."""

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, width: int, kernel_size: int, *, key):
        k1, k2 = jax.random.split(key)
        pad = kernel_size // 2
        self.conv1 = eqx.nn.Conv1d(width, width, kernel_size, padding=pad, key=k1)
        self.conv2 = eqx.nn.Conv1d(width, width, kernel_size, padding=pad, key=k2)

    def __call__(self, x):
        y = jax.nn.relu(self.conv1(x))
        y = self.conv2(y)
        return jax.nn.relu(x + y)


class AlphaModel(eqx.Module):
    """Conv(in, w) -> ResBlock(w)^depth -> Conv(w, 1) -> sigmoid."""

    lift: eqx.nn.Conv1d
    blocks: tuple
    head: eqx.nn.Conv1d

    def __init__(self, in_channels: int, width: int = 16, kernel_size: int = 3,
                 depth: int = 1, *, key, head_bias: float = -3.0):
        keys = jax.random.split(key, depth + 2)
        pad = kernel_size // 2
        self.lift = eqx.nn.Conv1d(in_channels, width, kernel_size, padding=pad,
                                  key=keys[0])
        self.blocks = tuple(
            ResBlock(width, kernel_size, key=keys[1 + i]) for i in range(depth))
        head = eqx.nn.Conv1d(width, 1, kernel_size, padding=pad, key=keys[-1])
        # Zero kernel + negative bias: the untrained policy outputs the
        # constant sigmoid(head_bias) ~ 0.05, i.e. an almost-pure DG scheme
        # that keeps the solver stable at the start of training (same trick
        # as the paper's softplus(-3) viscosity init).
        head = eqx.tree_at(lambda m: m.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda m: m.bias, head,
                           jnp.full_like(head.bias, head_bias))
        self.head = head

    def __call__(self, features):
        """features: (in_channels, n_elem) -> alpha (n_elem,) in (0, 1)."""
        x = jax.nn.relu(self.lift(features))
        for block in self.blocks:
            x = block(x)
        return jax.nn.sigmoid(self.head(x))[0]


def save_model(path: str, model: AlphaModel):
    eqx.tree_serialise_leaves(path, model)


def load_model(path: str, template: AlphaModel) -> AlphaModel:
    """template: a model built with the same hyperparameters (any key)."""
    return eqx.tree_deserialise_leaves(path, template)
