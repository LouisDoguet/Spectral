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
    
class Block(eqx.Module):
    """Two same-width convolutions (Fig. 5 inner block)."""

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, width: int, kernel_size: int, *, key):
        k1, k2 = jax.random.split(key)
        pad = kernel_size // 2
        self.conv1 = eqx.nn.Conv1d(width, width, kernel_size, padding=pad, key=k1)
        self.conv2 = eqx.nn.Conv1d(width, width, kernel_size, padding=pad, key=k2)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        return jax.nn.relu(self.conv2(x))


class AlphaModel(eqx.Module):
    """Conv(in, w) -> ResBlock(w)^depth -> Conv(w, 1) -> sigmoid."""

    lift: eqx.nn.Conv1d
    blocks: tuple
    head: eqx.nn.Conv1d

    def __init__(self, in_channels: int, width: int = 16, kernel_size: int = 3,
                 depth: int = 1, *, key, alpha_init: float = 0.05,
                 stable_init: bool = True, bool_res:bool = True):
        keys = jax.random.split(key, depth + 2)
        pad = kernel_size // 2
        self.lift = eqx.nn.Conv1d(in_channels, width, kernel_size, padding=pad,
                                  key=keys[0])
        if bool_res:
            self.blocks = tuple(ResBlock(width, kernel_size, key=keys[1 + i]) for i in range(depth))
        else:
            self.blocks = tuple(Block(width, kernel_size, key=keys[1 + i]) for i in range(depth))

        head = eqx.nn.Conv1d(width, 1, kernel_size, padding=pad, key=keys[-1])

        if stable_init:
            # Zero kernel + negative bias: the untrained policy outputs the
            # constant sigmoid(head_bias) ~ alpha_init, an almost-pure DG scheme
            # that keeps the solver stable at the start of training. (For
            # supervised PP-pretraining use stable_init=False: this saturated
            # readout kills gradients.)
            head_bias = float(jnp.log(alpha_init / (1.0 - alpha_init)))
            head = eqx.tree_at(lambda m: m.weight, head,
                               jnp.zeros_like(head.weight))
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

    Input  : (2 + Nn, n_elem*Nn)  -- per node, channels 0..1 are scalar
             indicators (the PP alpha baseline + the DG-FV density residual; see
             network.policy.alpha_features), channels 2..Nn+1 are the one-hot
             position within the element (subcell-position awareness). The
             spatial axis is the GLOBAL node sequence, so the convolution sees
             neighbouring nodes AND neighbouring elements. The PP channel lets
             the network start exactly at Persson-Peraire and learn a subcell
             correction from the residual + position.
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
    precondition: bool = eqx.field(static=True)

    def __init__(self, P: int, width: int = 16, kernel_size: int = 3,
                 depth: int = 1, *, key, alpha_init: float = 0.05,
                 stable_init: bool = True, n_data_channels: int = 2, bool_res: bool = True,
                 precondition: bool = False):
        Nn = P + 1
        # input = n_data_channels data rows + Nn one-hot position rows. The
        # data-channel count comes from network.policy.NODAL_DATA_CHANNELS.
        in_channels = n_data_channels + Nn
        keys = jax.random.split(key, depth + 2)
        pad = kernel_size // 2
        self.lift = eqx.nn.Conv1d(in_channels, width, kernel_size, padding=pad,
                                  key=keys[0])
        if bool_res:
            self.blocks = tuple(ResBlock(width, kernel_size, key=keys[1 + i]) for i in range(depth))
        else:
            self.blocks = tuple(Block(width, kernel_size, key=keys[1 + i]) for i in range(depth))

        if stable_init:
            # Zero readout weight + logit(alpha_init) bias: the untrained policy
            # outputs the constant alpha_init (almost pure DG) so the solver is
            # stable at the start of Stage-2 training.
            self.proj_w = jnp.zeros((width,))
            self.proj_b = jnp.array(jnp.log(alpha_init / (1.0 - alpha_init)))
        else:
            # Trainable readout for supervised PP-pretraining (a saturated
            # readout would kill the regression gradients).
            self.proj_w = 0.1 * jax.random.normal(keys[-1], (width,))
            self.proj_b = jnp.array(0.0)
        self.Nn = Nn
        self.precondition = precondition

    def __call__(self, features):
        """features: (n_data_channels + Nn, n_elem*Nn) -> alpha (n_elem, P).

        When self.precondition is set, the DATA channels (everything but the Nn
        one-hot rows) are whitened as an (N_points, N_features) matrix before the
        one-hot block is re-appended -- see whiten_data_channels."""
        if self.precondition:
            n_data = features.shape[0] - self.Nn
            data, onehot = features[:n_data], features[n_data:]   # strip one-hot
            data = whiten_rows(data)                              # precondition
            features = jnp.concatenate([data, onehot], axis=0)    # re-append one-hot
        x = jax.nn.relu(self.lift(features))
        for block in self.blocks:
            x = block(x)                       # (width, n_elem*Nn)
        width = x.shape[0]
        h = x.T.reshape(-1, self.Nn, width)    # (n_elem, Nn, width)
        hi = 0.5 * (h[:, :-1, :] + h[:, 1:, :])  # (n_elem, P, width) interface feats
        logit = hi @ self.proj_w + self.proj_b   # (n_elem, P)
        return jax.nn.sigmoid(logit)


def whiten_rows(rows, eps: float = 1e-5):
    """Per-sample ZCA whitening of a feature matrix, used to *precondition* the
    network input (toggled by NodalAlphaModel.precondition).

    rows: (F, N) = (N_features, N_integration_points). Interpreted as the feature
    space X = rows.T of shape (N_points, N_features): each integration point is a
    sample, each channel a feature. Returns the whitened rows, same shape (F, N),
    with the transformed features decorrelated and unit-variance (identity
    covariance up to the eps ridge):

        Xc = X - mean_over_points(X)
        Σ  = (Xc^T Xc)/N + eps I            # (F, F) feature covariance
        Xw = Xc @ Σ^(-1/2)                  # symmetric (ZCA) inverse sqrt

    Statistics are computed from THIS state's own points (instance whitening), so
    it is stateless and differentiable. A small eps keeps Σ^(-1/2) well-defined
    when a feature is (near-)constant or two features are collinear."""
    X = rows.T                                   # (N_points, N_features)
    n, f = X.shape
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / n + eps * jnp.eye(f)     # (F, F), symmetric PSD
    evals, evecs = jnp.linalg.eigh(cov)          # cov = V diag(evals) V^T
    inv_sqrt = (evecs * (1.0 / jnp.sqrt(evals))) @ evecs.T   # Σ^(-1/2)
    return (Xc @ inv_sqrt).T                      # back to (F, N)


def nodal_features(scalars):
    """Build the nodal network input from per-node scalar channels.

    scalars: (C, n_elem, Nn) -> (C + Nn, n_elem*Nn). The first C rows are the
    scalar channels over the GLOBAL node sequence; the last Nn rows are the
    one-hot of the node's position within its element. Pure array-shaping (no DG
    knowledge) -- the caller (network.policy.alpha_features) decides the scalars.
    Built with jnp so gradients flow through the scalar channels."""
    C, n_elem, Nn = scalars.shape
    rows = scalars.reshape(C, -1)                # (C, n_elem*Nn)
    onehot = jnp.tile(jnp.eye(Nn), (1, n_elem))  # (Nn, n_elem*Nn)
    return jnp.concatenate([rows, onehot], axis=0)


# ---------------------------------------------------------------------------
# PNO + GNN sensor (P-independent, replaces the CNN + one-hot pipeline)
# ---------------------------------------------------------------------------

def apply_linear(lin, x):
    """Apply an eqx.nn.Linear over the LAST axis of an arbitrarily-shaped x.

    eqx.nn.Linear.__call__ only takes a single (in,) vector; this is the shared
    per-point / per-mode application (same weights at every position), which is
    exactly what keeps the graph model independent of P and n_elem."""
    y = x @ lin.weight.T
    return y if lin.bias is None else y + lin.bias


class PolynomialNeuralOperator(eqx.Module):
    """Modal energy -> (global element feature, nodal reconstruction), any P.

    The P-independence mechanism: each mode k becomes a token
    (coord_k, log-energy_k) with coord_k = k/P the normalized frequency (the
    modal-space analogue of the point coordinate xi), a SHARED per-mode MLP maps
    every token to `channels` features (fixed parameter count for any number of
    modes), and
      1. mean-pooling over modes gives the global feature (n_elem, channels);
      2. the Legendre Vandermonde Phi (mesh constant, a formula -- not learned)
         projects the mode features back onto the integration points:
         nodal = Phi @ mode_feat, (n_elem, Nn, channels).
    Mode ORDER is kept (mode index = frequency, physically meaningful); only
    the point/element axes are permutation-safe.

    The log10 transform is the (formula-based) preconditioning of the energy:
    the normalized spectrum spans ~[1e-12, 1] and is untrainable raw (same
    rationale as the PP decision rescaling in network.policy.channel_energy)."""

    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    channels: int = eqx.field(static=True)

    def __init__(self, hidden: int, channels: int, *, key):
        k1, k2 = jax.random.split(key)
        self.lin1 = eqx.nn.Linear(2, hidden, key=k1)
        self.lin2 = eqx.nn.Linear(hidden, channels, key=k2)
        self.channels = channels

    def __call__(self, energy, Phi):
        """energy: (n_elem, Nn) per-element spectrum, Phi: (Nn, Nn) Vandermonde
        -> (pno_global (n_elem, C), nodal (n_elem, Nn, C))."""
        n_elem, Nn = energy.shape
        coord = jnp.arange(Nn) / max(Nn - 1, 1)             # k/P, the P-trick
        e_log = jnp.log10(energy + 1e-12) / 12.0            # ~[-1, 0]
        tokens = jnp.stack([jnp.broadcast_to(coord, (n_elem, Nn)), e_log],
                           axis=-1)                          # (n_elem, Nn, 2)
        mode_feat = apply_linear(self.lin2,
                                 jax.nn.relu(apply_linear(self.lin1, tokens)))
        pno_global = jnp.mean(mode_feat, axis=1)             # (n_elem, C)
        nodal = jnp.einsum("ik,ekc->eic", Phi, mode_feat)    # (n_elem, Nn, C)
        return pno_global, nodal


class GraphBlock(eqx.Module):
    """One volume + one surface message-passing round, both residual.

    The graph is the ACTUAL DGSEM coupling, in dense form (no edge lists):
      - volume: every intra-element point pair, weighted by the differentiation
        matrix -- receiver j aggregates sum_k D[j,k] m_k, which is exactly
        einsum('jk,ekf->ejf', D, m), the same contraction the split-form volume
        term uses (jax_dgsem.solver._subcell_flux_dg);
      - surface: the two face nodes exchange with the touching neighbour face
        node (roll over the element axis; wrap rows zeroed when not periodic),
        the graph analogue of the numerical-flux exchange the old per-element
        CNN could not represent.
    No order-dependent op (no conv, no reshape-to-grid): position enters ONLY
    through D, xi and the face pairing, i.e. the physical structure.

    D must be pre-normalized by the caller (GraphAlphaModel.point_latents
    divides by P(P+1), D's row-sum growth rate) -- the raw reference-element D
    is unbounded in P (row-sums ~16/37/68 at P=4/6/8), which would make the
    volume round blow up at high P and dominate the real (state-dependent)
    signal with a P-and-position-only bias at any fixed P."""

    vol_msg: eqx.nn.Linear
    vol_upd: eqx.nn.Linear
    surf_msg: eqx.nn.Linear
    surf_upd: eqx.nn.Linear

    def __init__(self, hidden: int, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.vol_msg = eqx.nn.Linear(hidden, hidden, key=k1)
        self.vol_upd = eqx.nn.Linear(2 * hidden, hidden, key=k2)
        self.surf_msg = eqx.nn.Linear(hidden, hidden, key=k3)
        self.surf_upd = eqx.nn.Linear(2 * hidden, hidden, key=k4)

    def __call__(self, h, D, periodic: bool):
        """h: (n_elem, Nn, H) -> (n_elem, Nn, H)."""
        # volume round: D-weighted intra-element aggregation
        m = jax.nn.relu(apply_linear(self.vol_msg, h))
        agg = jnp.einsum("jk,ekf->ejf", D, m)
        h = jax.nn.relu(h + apply_linear(self.vol_upd,
                                         jnp.concatenate([h, agg], axis=-1)))
        # surface round: face-node exchange with the neighbour element
        m = jax.nn.relu(apply_linear(self.surf_msg, h))
        from_left = jnp.roll(m[:, -1, :], 1, axis=0)    # elem e <- (e-1)'s last
        from_right = jnp.roll(m[:, 0, :], -1, axis=0)   # elem e <- (e+1)'s first
        if not periodic:
            from_left = from_left.at[0].set(0.0)        # no neighbour outside
            from_right = from_right.at[-1].set(0.0)
        agg = (jnp.zeros_like(h).at[:, 0, :].set(from_left)
               .at[:, -1, :].set(from_right))
        return jax.nn.relu(h + apply_linear(self.surf_upd,
                                            jnp.concatenate([h, agg], axis=-1)))


class DGSEMGraphNet(eqx.Module):
    """Encoder + `depth` GraphBlocks on the DGSEM graph: (n_elem, Nn, in) ->
    (n_elem, Nn, hidden). Permutation-equivariant over points and elements."""

    encoder: eqx.nn.Linear
    blocks: tuple

    def __init__(self, in_features: int, hidden: int, depth: int, *, key):
        keys = jax.random.split(key, depth + 1)
        self.encoder = eqx.nn.Linear(in_features, hidden, key=keys[0])
        self.blocks = tuple(GraphBlock(hidden, key=keys[1 + i])
                            for i in range(depth))

    def __call__(self, node_feats, D, periodic: bool):
        h = jax.nn.relu(apply_linear(self.encoder, node_feats))
        for blk in self.blocks:
            h = blk(h, D, periodic)
        return h


class GraphAlphaModel(eqx.Module):
    """PNO + GNN alpha policy -- the P- and n_elem-independent sensor.

    Input  : features (2, n_elem, Nn) = stacked [DG-FV density residual per
             node, modal-energy spectrum per element] (network.policy
             .graph_features -- NO one-hot), plus the mesh (whose quads/D/Phi
             are the reference-element constants the branches reuse; passed as
             an argument so they never enter the trainable pytree).
    Output : (alpha, alpha_boundary).
             alpha: (n_elem, P) in (0,1) per interior subcell interface -- the
             same contract as NodalAlphaModel, formed by the same adjacent-
             node-pair readout, consumed by hybrid_residual's subcell blend.
             alpha_boundary: (n_elem,) in (0,1) per TRUE element-to-element
             interface, formed by the SAME readout applied across the element
             roll instead of within an element -- consumed by
             hybrid_residual's alpha_boundary argument to blend the
             element-interface flux itself (entropy-conservative vs.
             entropy-stable-LF), the one coupling the subcell alpha can never
             reach.

    Forward: PNO(energy) -> global + nodal features; node features =
    [residual, xi, nodal] (xi = mesh.quads replaces the one-hot); GNN message
    passing; fuse with the broadcast global feature; pointwise MLP; interface
    readout (subcell AND element-boundary, same weights). Unlike
    NodalAlphaModel this runs unchanged at ANY P: no weight shape depends on
    Nn."""

    pno: PolynomialNeuralOperator
    gnn: DGSEMGraphNet
    fuse: eqx.nn.Linear
    proj_w: jnp.ndarray            # (fusion_hidden,) interface readout weight
    proj_b: jnp.ndarray            # scalar bias

    def __init__(self, pno_channels: int = 8, pno_hidden: int = 32,
                 gnn_hidden: int = 24, fusion_hidden: int = 32, depth: int = 1,
                 *, key, alpha_init: float = 0.05, stable_init: bool = True):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.pno = PolynomialNeuralOperator(pno_hidden, pno_channels, key=k1)
        self.gnn = DGSEMGraphNet(2 + pno_channels, gnn_hidden, depth, key=k2)
        self.fuse = eqx.nn.Linear(gnn_hidden + pno_channels, fusion_hidden,
                                  key=k3)
        if stable_init:
            # Zero readout weight + logit(alpha_init) bias: the untrained
            # policy outputs the constant alpha_init (almost pure DG) so the
            # solver is stable at the start of Stage-2 training.
            self.proj_w = jnp.zeros((fusion_hidden,))
            self.proj_b = jnp.array(jnp.log(alpha_init / (1.0 - alpha_init)))
        else:
            # Trainable readout for supervised PP-pretraining.
            self.proj_w = 0.1 * jax.random.normal(k4, (fusion_hidden,))
            self.proj_b = jnp.array(0.0)

    def point_latents(self, features, mesh):
        """Per-point latents (n_elem, Nn, fusion_hidden) -- everything before
        the interface pairing; permutation-equivariant (tested)."""
        residual, energy = features[0], features[1]      # (n_elem, Nn) each
        n_elem, Nn = residual.shape
        pno_global, nodal = self.pno(energy, mesh.Phi)
        xi = jnp.broadcast_to(mesh.quads, (n_elem, Nn))
        node_feats = jnp.concatenate(
            [residual[..., None], xi[..., None], nodal], axis=-1)
        # D's row-sums grow like P(P+1) (the GLL differentiation matrix is a
        # reference-element operator, not scaled by dx like the physical
        # derivative is) -- normalize so the volume round stays an O(1)
        # aggregation at any P; this is what makes the SAME trained weights
        # actually behave the same at P=3 and P=8 instead of blowing up at
        # high P (unnormalized row-sums: ~16 at P=4, ~37 at P=6, ~68 at P=8).
        D_msg = mesh.D / (mesh.P * (mesh.P + 1))
        h = self.gnn(node_feats, D_msg, mesh.periodic)
        fused = jnp.concatenate(
            [h, jnp.broadcast_to(pno_global[:, None, :],
                                 (n_elem, Nn, self.pno.channels))], axis=-1)
        return jax.nn.relu(apply_linear(self.fuse, fused))

    def __call__(self, features, mesh):
        """features (2, n_elem, Nn), mesh -> (alpha, alpha_boundary).

        alpha: (n_elem, P) in (0, 1), one per interior subcell interface (the
        existing contract, unchanged).
        alpha_boundary: (n_elem,) in (0, 1), one per TRUE element-to-element
        interface (element e's right face, shared with element e+1's left
        face) -- consumed by jax_dgsem.solver.hybrid_residual to blend the
        interface flux itself, the one place the subcell alpha above can
        never reach (see solver._apply_surface_and_mass_inverse).

        Deliberately reuses the SAME proj_w/proj_b readout as the subcell
        interfaces rather than a second head: an interior interface latent is
        the average of its two adjacent NODE latents (z[:,:-1]+z[:,1:]); the
        boundary interface is the exact same averaging, just paired across
        the element roll (last node of e with first node of e+1) instead of
        within-element indices. One mechanism, one set of weights, for every
        integration point -- element edges are not a structurally different
        case for the readout, only for which two node latents feed it."""
        z = self.point_latents(features, mesh)
        zi = 0.5 * (z[:, :-1, :] + z[:, 1:, :])   # (n_elem, P) interface feats
        logit = zi @ self.proj_w + self.proj_b
        alpha = jax.nn.sigmoid(logit)

        z_bound = 0.5 * (z[:, -1, :] + jnp.roll(z[:, 0, :], -1, axis=0))
        logit_b = z_bound @ self.proj_w + self.proj_b
        alpha_boundary = jax.nn.sigmoid(logit_b)
        return alpha, alpha_boundary


def save_model(path: str, model):
    eqx.tree_serialise_leaves(path, model)


def load_model(path: str, template):
    """template: a model built with the same hyperparameters (any key)."""
    return eqx.tree_deserialise_leaves(path, template)