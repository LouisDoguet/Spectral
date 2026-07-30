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
# Element-GNN -> OPNO sensor (P- and n_elem-independent, the inverted pipeline)
# ---------------------------------------------------------------------------

# Floor for the log-energy transform. log10(clip(e, floor, 1)) is identical to
# log10(e) for every mode that carries real signal (PP's own threshold at P=6
# is ~6e-4) but has EXACTLY zero gradient below the floor. The GNN_PNO branch
# used log10(e + 1e-12), whose backward factor at floor-level energies is
# ~4e11: bounded forward, exploding backward, and it sits inside the training
# closed loop -- measured there as the dominant cause of the 1e28-1e33 BPTT
# gradients. Round-off-level mode energies are noise; their gradient SHOULD
# be zero.
ENERGY_FLOOR = 1e-6

# Fixed physical scales of the asinh residual channels (network.policy
# .opno_features). Deliberately NOT state-dependent (no max/std normalization):
# a state-dependent scale either crushes smooth-region signal under a global
# max (the alpha-comb failure mode on GNN_PNO) or re-amplifies round-off under
# a per-element max (measured worse ||dalpha/dU||). asinh is linear at small
# arguments and logarithmic at large ones, so two fixed scales cover the
# smooth-to-shock dynamic range with a gradient bounded by 1/s.
RES_SCALES = (1.0, 100.0)


def apply_linear(lin, x):
    """Apply an eqx.nn.Linear over the LAST axis of an arbitrarily-shaped x.

    eqx.nn.Linear.__call__ only takes a single (in,) vector; this is the shared
    per-token application (same weights at every node/mode/element), which is
    what keeps the opno model independent of P and n_elem."""
    y = x @ lin.weight.T
    return y if lin.bias is None else y + lin.bias


def log_energy(energy):
    """Bounded-backward log transform of a normalized modal spectrum, ~[-1, 0]."""
    return jnp.log10(jnp.clip(energy, ENERGY_FLOOR, 1.0)) / 6.0


class TokenMLP(eqx.Module):
    """Two shared Linear layers applied over the last axis: the per-token map
    every P-independent branch reuses (tokens = nodes, modes, or elements)."""

    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self, n_in: int, hidden: int, n_out: int, *, key):
        k1, k2 = jax.random.split(key)
        self.lin1 = eqx.nn.Linear(n_in, hidden, key=k1)
        self.lin2 = eqx.nn.Linear(hidden, n_out, key=k2)

    def __call__(self, x):
        return apply_linear(self.lin2, jax.nn.relu(apply_linear(self.lin1, x)))


class ElementEncoder(eqx.Module):
    """Per-element embedding from (residual nodes, modal spectrum), any P.

    Two token-pooling branches (the P-trick: shared per-token MLP + mean pool,
    so no weight shape depends on Nn):
      - residual tokens (xi_i, r1_i, r2_i) over the element's nodes -- xi is a
        token COORDINATE here (inside a pooled sum), not a raw feature channel
        the readout could latch onto;
      - spectrum tokens (k/P, logE_k) over the element's modes (order kept:
        mode index = frequency).
    Concatenated and projected to the GNN width."""

    res_mlp: TokenMLP
    spec_mlp: TokenMLP
    proj: eqx.nn.Linear

    def __init__(self, n_res_channels: int, hidden: int, width: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.res_mlp = TokenMLP(1 + n_res_channels, hidden, hidden, key=k1)
        self.spec_mlp = TokenMLP(2, hidden, hidden, key=k2)
        self.proj = eqx.nn.Linear(2 * hidden, width, key=k3)

    def __call__(self, res_channels, energy, quads):
        """res_channels (C_res, n_elem, Nn), energy (n_elem, Nn), quads (Nn,)
        -> (n_elem, width)."""
        n_res, n_elem, Nn = res_channels.shape
        xi = jnp.broadcast_to(quads, (n_elem, Nn))
        res_tok = jnp.concatenate(
            [xi[..., None], jnp.moveaxis(res_channels, 0, -1)], axis=-1)
        res_emb = jnp.mean(self.res_mlp(res_tok), axis=1)      # (n_elem, hidden)

        coord = jnp.arange(Nn) / max(Nn - 1, 1)                # k/P
        spec_tok = jnp.stack(
            [jnp.broadcast_to(coord, (n_elem, Nn)), log_energy(energy)], axis=-1)
        spec_emb = jnp.mean(self.spec_mlp(spec_tok), axis=1)   # (n_elem, hidden)

        return apply_linear(self.proj,
                            jnp.concatenate([res_emb, spec_emb], axis=-1))


class ElementGraphNet(eqx.Module):
    """Message passing on the element line graph: how neighbouring elements'
    features relate. Nodes are ELEMENTS (not integration points -- the
    inversion from the GNN_PNO design); edges are the physical adjacency the
    numerical flux couples, in dense roll form (no edge lists, jit-friendly,
    n_elem-independent). Direction-aware: separate left/right message MLPs
    (upwind and downwind neighbours are physically different)."""

    msg_l: eqx.nn.Linear
    msg_r: eqx.nn.Linear
    upd: eqx.nn.Linear

    def __init__(self, width: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.msg_l = eqx.nn.Linear(width, width, key=k1)
        self.msg_r = eqx.nn.Linear(width, width, key=k2)
        self.upd = eqx.nn.Linear(3 * width, width, key=k3)

    def __call__(self, h, periodic: bool):
        """h (n_elem, width) -> (n_elem, width), one residual round."""
        m_l = jax.nn.relu(apply_linear(self.msg_l, h))   # what e sends rightward
        m_r = jax.nn.relu(apply_linear(self.msg_r, h))   # what e sends leftward
        from_left = jnp.roll(m_l, 1, axis=0)             # e receives from e-1
        from_right = jnp.roll(m_r, -1, axis=0)           # e receives from e+1
        if not periodic:
            from_left = from_left.at[0].set(0.0)         # no neighbour outside
            from_right = from_right.at[-1].set(0.0)
        agg = jnp.concatenate([h, from_left, from_right], axis=-1)
        return jax.nn.relu(h + apply_linear(self.upd, agg))


class ModalOPNO(eqx.Module):
    """Orthogonal-polynomial neural operator decoder: enrich each mode of the
    spectrum with the element's neighbour-aware GNN latent, then reconstruct
    per-node features through the Legendre Vandermonde.

    Per-mode tokens (k/P, logE_k, h_e) -> shared MLP -> mode features
    (n_elem, Nn, C); nodal = Phi @ mode features. Phi is a mesh constant (a
    formula, not learned), which is what makes the reconstruction run at any
    P; the within-element alpha profile is therefore a smooth Legendre
    expansion by construction. Mode ORDER is kept (mode index = frequency)."""

    mlp: TokenMLP
    channels: int = eqx.field(static=True)

    def __init__(self, width: int, hidden: int, channels: int, *, key):
        self.mlp = TokenMLP(2 + width, hidden, channels, key=key)
        self.channels = channels

    def __call__(self, energy, h, Phi):
        """energy (n_elem, Nn), h (n_elem, width), Phi (Nn, Nn)
        -> mode feats (n_elem, Nn, C), nodal feats (n_elem, Nn, C)."""
        n_elem, Nn = energy.shape
        coord = jnp.arange(Nn) / max(Nn - 1, 1)
        tokens = jnp.concatenate(
            [jnp.broadcast_to(coord, (n_elem, Nn))[..., None],
             log_energy(energy)[..., None],
             jnp.broadcast_to(h[:, None, :], (n_elem, Nn, h.shape[-1]))],
            axis=-1)
        mode_feat = self.mlp(tokens)                         # (n_elem, Nn, C)
        nodal = jnp.einsum("ik,ekc->eic", Phi, mode_feat)    # (n_elem, Nn, C)
        return mode_feat, nodal


class OPNOAlphaModel(eqx.Module):
    """Element-GNN -> OPNO alpha policy (the inverted, P-/n_elem-independent
    sensor).

    Input  : features (2 + len(RES_SCALES) - 1, n_elem, Nn) from
             network.policy.opno_features = stacked [asinh residual channels
             (node axis), modal-energy spectrum (MODE axis)], plus the mesh
             (quads/Phi/periodic are reference-element constants passed as an
             argument so they never enter the trainable pytree).
    Output : alpha (n_elem, P) in (0,1) per interior subcell interface -- the
             exact NodalAlphaModel contract, consumed unchanged by
             jax_dgsem.solver.hybrid_residual. (No element-interface blend on
             this branch: the solver keeps its validated entropy-stable-LF
             interfaces.)

    Forward: ElementEncoder pools each element's residual nodes and spectrum
    modes into h0; ElementGraphNet exchanges depth rounds of neighbour
    messages (the "how elements relate" stage); ModalOPNO re-expands the
    spectrum, mode by mode, conditioned on the neighbour-aware latent, and
    reconstructs nodal features through Phi; a pointwise fuse combines
    [nodal reconstruction, per-node residual channels, h_e] so subcell
    localization survives the element-level pooling; the interface readout is
    the same adjacent-node averaging as NodalAlphaModel."""

    encoder: ElementEncoder
    gnn_blocks: tuple
    opno: ModalOPNO
    fuse: eqx.nn.Linear
    proj_w: jnp.ndarray            # (fusion_hidden,) interface readout weight
    proj_b: jnp.ndarray            # scalar bias
    n_res: int = eqx.field(static=True)

    def __init__(self, n_res_channels: int = 2, width: int = 24,
                 opno_hidden: int = 32, opno_channels: int = 8,
                 fusion_hidden: int = 32, depth: int = 2, *, key,
                 alpha_init: float = 0.05, stable_init: bool = True):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.encoder = ElementEncoder(n_res_channels, opno_hidden, width, key=k1)
        self.gnn_blocks = tuple(
            ElementGraphNet(width, key=k) for k in jax.random.split(k2, depth))
        self.opno = ModalOPNO(width, opno_hidden, opno_channels, key=k3)
        self.fuse = eqx.nn.Linear(opno_channels + n_res_channels + width,
                                  fusion_hidden, key=k4)
        if stable_init:
            # Zero readout weight + logit(alpha_init) bias: the untrained
            # policy outputs the constant alpha_init (almost pure DG) so the
            # solver is stable at the start of Stage-2 training.
            self.proj_w = jnp.zeros((fusion_hidden,))
            self.proj_b = jnp.array(jnp.log(alpha_init / (1.0 - alpha_init)))
        else:
            # Trainable readout for supervised PP-pretraining (a saturated
            # readout would kill the regression gradients).
            self.proj_w = 0.1 * jax.random.normal(k5, (fusion_hidden,))
            self.proj_b = jnp.array(0.0)
        self.n_res = n_res_channels

    def element_latents(self, features, mesh):
        """The neighbour-aware element latents h (n_elem, width) after the GNN
        -- Stage A + B, exposed for the tests (pooling invariance, locality)."""
        res_channels, energy = features[:self.n_res], features[self.n_res]
        h = self.encoder(res_channels, energy, mesh.quads)
        for blk in self.gnn_blocks:
            h = blk(h, mesh.periodic)
        return h

    def point_latents(self, features, mesh):
        """Per-node latents (n_elem, Nn, fusion_hidden): everything before the
        interface pairing."""
        res_channels, energy = features[:self.n_res], features[self.n_res]
        n_elem, Nn = energy.shape
        h = self.element_latents(features, mesh)
        _, nodal = self.opno(energy, h, mesh.Phi)
        fused = jnp.concatenate(
            [nodal, jnp.moveaxis(res_channels, 0, -1),
             jnp.broadcast_to(h[:, None, :], (n_elem, Nn, h.shape[-1]))],
            axis=-1)
        return jax.nn.relu(apply_linear(self.fuse, fused))

    def __call__(self, features, mesh):
        """features (1+n_res, n_elem, Nn), mesh -> alpha (n_elem, P)."""
        z = self.point_latents(features, mesh)
        zi = 0.5 * (z[:, :-1, :] + z[:, 1:, :])   # (n_elem, P, fusion_hidden)
        logit = zi @ self.proj_w + self.proj_b    # (n_elem, P)
        return jax.nn.sigmoid(logit)


def save_model(path: str, model):
    eqx.tree_serialise_leaves(path, model)


def load_model(path: str, template):
    """template: a model built with the same hyperparameters (any key)."""
    return eqx.tree_deserialise_leaves(path, template)