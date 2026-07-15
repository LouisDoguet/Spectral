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
                 depth: int = 1, *, key, alpha_init: float = 0.05):
        head_bias = float(jnp.log(alpha_init / (1.0 - alpha_init)))
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


class NodalAlphaModel(eqx.Module):
    """Nodal / subcell alpha policy.

    Input  : (1 + Nn, n_elem*Nn)  -- per node, channel 0 = density, channels
             1..Nn = one-hot position within the element (subcell-position
             awareness). The spatial axis is the GLOBAL node sequence, so the
             convolution sees neighbouring nodes AND neighbouring elements.
    Output : (n_elem, P)  -- one alpha in (0,1) per interior subcell interface,
             formed by combining each adjacent node pair's latent features.
             This is exactly the shape the flux-blend hybrid_residual consumes.

    Still fully convolutional over the node axis, so it runs on any n_elem; it
    is tied to a fixed P (the one-hot width and the interface count depend on
    Nn = P+1)."""

    lift: eqx.nn.Conv1d
    blocks: tuple
    proj_w: jnp.ndarray            # (width,) interface readout weight
    proj_b: jnp.ndarray            # scalar bias
    Nn: int = eqx.field(static=True)

    def __init__(self, P: int, width: int = 16, kernel_size: int = 3,
                 depth: int = 1, *, key, alpha_init: float = 0.05):
        Nn = P + 1
        in_channels = 1 + Nn
        keys = jax.random.split(key, depth + 1)
        pad = kernel_size // 2
        self.lift = eqx.nn.Conv1d(in_channels, width, kernel_size, padding=pad,
                                  key=keys[0])
        self.blocks = tuple(
            ResBlock(width, kernel_size, key=keys[1 + i]) for i in range(depth))
        # Zero readout weight + logit(alpha_init) bias: the untrained policy
        # outputs the constant alpha_init ~ 0.05 (almost pure DG) so the solver
        # is stable at the start of training.
        self.proj_w = jnp.zeros((width,))
        self.proj_b = jnp.array(jnp.log(alpha_init / (1.0 - alpha_init)))
        self.Nn = Nn

    def __call__(self, features):
        """features: (1+Nn, n_elem*Nn) -> alpha (n_elem, P) in (0, 1)."""
        x = jax.nn.relu(self.lift(features))
        for block in self.blocks:
            x = block(x)                       # (width, n_elem*Nn)
        width = x.shape[0]
        h = x.T.reshape(-1, self.Nn, width)    # (n_elem, Nn, width)
        hi = 0.5 * (h[:, :-1, :] + h[:, 1:, :])  # (n_elem, P, width) interface feats
        logit = hi @ self.proj_w + self.proj_b   # (n_elem, P)
        return jax.nn.sigmoid(logit)


def nodal_features(U, mesh):
    """Network input for NodalAlphaModel: (1 + Nn, n_elem*Nn).

    Channel 0 is the nodal density; channels 1..Nn are a one-hot of the node's
    position within its element (repeats every Nn nodes). Built with jnp so
    gradients flow through the density channel."""
    rho = U[0]                                 # (n_elem, Nn)
    n_elem, Nn = rho.shape
    rho_row = rho.reshape(1, -1)               # (1, n_elem*Nn)
    onehot = jnp.tile(jnp.eye(Nn), (1, n_elem))  # (Nn, n_elem*Nn)
    return jnp.concatenate([rho_row, onehot], axis=0)


def save_model(path: str, model):
    eqx.tree_serialise_leaves(path, model)


def load_model(path: str, template):
    """template: a model built with the same hyperparameters (any key)."""
    return eqx.tree_deserialise_leaves(path, template)
