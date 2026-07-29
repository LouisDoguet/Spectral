"""Routing between the alpha policies (per-element, nodal/subcell, graph), and
the definition of the NETWORK INPUTS.

  "element": legacy per-element policy. Input = normalized modal-energy
             spectrum (P+1 channels) over the elements; output alpha (n_elem,).
  "nodal"  : per interior subcell interface. Input = a stack of per-node DATA
             channels followed by the one-hot position block; output alpha
             (n_elem, P).
  "graph"  : PNO + GNN policy (network.model.GraphAlphaModel), P- and
             n_elem-independent. Input = graph_features (2, n_elem, Nn) =
             [residual per node, modal-energy spectrum per element] -- no
             one-hot (xi and the DGSEM graph replace it); the mesh is passed
             to the model alongside the features (its quads/D/Phi are the
             reference-element constants the branches reuse). Output alpha
             (n_elem, P). Call through apply_alpha / call_model, which hide
             the signature difference from the legacy models.

  Nodal input layout (this is the thing to tune):

      [ NODAL_DATA_CHANNELS... ,  one-hot position (Nn rows) ]

  To change the input, EDIT the NODAL_DATA_CHANNELS list below -- add or remove
  a (name, function) line. Each function maps (U, mesh) -> a per-node scalar
  field of shape (n_elem, Nn). Everything else (the model's in_channels, the
  saved meta) follows from the list length automatically, so reverting to an
  earlier input is just deleting a line. Changing the list makes previously
  trained checkpoints incompatible (different in_channels) -- retrain.
"""

import jax
import jax.numpy as jnp

from jax_dgsem.indicator import modal_energy, persson_peraire_indicator
from jax_dgsem.solver import dg_residual, fv_residual
from network.model import (AlphaModel, GraphAlphaModel, NodalAlphaModel,
                           nodal_features)


# ---------------------------------------------------------------------------
# Nodal data channels  (edit this list to change the input)
# ---------------------------------------------------------------------------

def channel_residual(U, mesh):
    """DG - FV density residual per node, normalized to ~[-1, 1]. The per-NODE
    troubled signal: where the high-order and robust operators disagree. This
    is what lets the network localize at the subcell level (beyond PP).

    The two edge nodes (index 0 and Nn-1) are excluded before normalizing and
    then overwritten with their nearest interior neighbour. Both builders set
    B_dg[0]==B_fv[0] and B_dg[-1]==B_fv[-1] exactly (the element-boundary flux
    is always the plain physical flux, never blended -- see
    jax_dgsem.solver._alpha_on_interfaces), so the raw residual there is a
    first difference against an EXACT zero, unlike every interior node's
    genuine second difference -- structurally ~5-15x larger, in EVERY state,
    smooth or shocked (verified: on a fully smooth mesh the single largest raw
    residual in the whole domain sits at an edge node of an element PP scores
    0.0, and even on a shocked mesh the argmax raw residual element is not the
    argmax-PP element). Left in, this artifact anchors the per-state max and
    teaches the network a position-only "high alpha near edges" shortcut that
    fires regardless of whether there is a real shock nearby."""
    res = (dg_residual(U, mesh) - fv_residual(U, mesh))[0]      # (n_elem, Nn)
    interior = res[:, 1:-1]
    res = res.at[:, 0].set(res[:, 1]).at[:, -1].set(res[:, -2])
    scale = jnp.max(jnp.abs(interior)) + 1e-8
    return res / scale


def channel_energy(U, mesh):
    """Persson-Peraire modal-energy indicator per ELEMENT, broadcast onto the
    element's nodes. Returned as PP's own smooth [0,1] "troubled" decision
    sigmoid((s/T)(E_ind - T)) rather than the raw E_ind: raw E_ind is tiny
    (~1e-6..1e-1) and PP thresholds it very steeply, so the network cannot learn
    that step from the raw value -- the pre-scaled decision is what actually
    lets the network reproduce PP."""
    P = mesh.P
    T = 0.5 * 10.0 ** (-1.8 * (P + 1) ** 0.25)          # PP threshold (P-dep.)
    s = 9.21024
    eind = persson_peraire_indicator(U, mesh)           # (n_elem,)
    decision = jax.nn.sigmoid((s / T) * (eind - T))     # (n_elem,) in [0,1]
    n_elem, Nn = U.shape[1], U.shape[2]
    return jnp.broadcast_to(decision[:, None], (n_elem, Nn))


def channel_modal_spectrum(U, mesh):
    """The FULL normalized density modal-energy spectrum per element -- the Nn =
    P+1 values feat_j = m_j^2 / sum_k m_k^2 (network.indicator.modal_energy) --
    transferred as P+1 data channels, each broadcast onto its element's nodes.

    This is the raw spectrum the scalar `energy` channel distills into a single
    PP decision; handing the network the whole spectrum lets it learn its own
    mode weighting instead of PP's fixed top-mode threshold. Note the spectrum
    sums to 1 across modes (the channels are collinear), so pair it with
    preconditioning (the eps ridge) or drop one mode if you whiten.

    Returns (Nn, n_elem, Nn) = (mode, elem, node)."""
    spec = modal_energy(U, mesh)                     # (n_elem, Nn) = (elem, mode)
    n_elem, Nn = spec.shape
    return jnp.broadcast_to(spec.T[:, :, None], (Nn, n_elem, Nn))


# Registry of every available data channel: name -> (fn, width). fn(U, mesh)
# returns (n_elem, Nn) when width == 1, else (width, n_elem, Nn); width is an int
# or a callable P -> int (the modal spectrum expands to P+1 channels).
CHANNELS = {
    "residual":  (channel_residual,       1),
    "energy":    (channel_energy,         1),
    "mspectrum": (channel_modal_spectrum, lambda P: P + 1),
}

# The ACTIVE nodal input for NEW training runs, in order (names from CHANNELS).
# Edit this list to change the input. It is recorded in every checkpoint's meta
# ("data_channels"), so a saved model is always evaluated on the channels it was
# trained with -- see channels_from_meta and alpha_features(channels=...). That
# lets old checkpoints (e.g. ["residual", "energy"]) be compared even after the
# default here changes.
NODAL_DATA_CHANNELS = ["residual", "mspectrum"]


def _channel_width(name, P):
    w = CHANNELS[name][1]
    return w(P) if callable(w) else w


def _resolve_channels(channels):
    return list(NODAL_DATA_CHANNELS if channels is None else channels)


def n_nodal_data(P: int, channels=None) -> int:
    """Total data-channel count for a channel set (some entries expand with P)."""
    return sum(_channel_width(nm, P) for nm in _resolve_channels(channels))


def nodal_channel_names(P: int, channels=None):
    """Per-channel names, expanding multi-channel entries; len == n_nodal_data."""
    names = []
    for nm in _resolve_channels(channels):
        k = _channel_width(nm, P)
        names += [nm] if k == 1 else [f"{nm}[{j}]" for j in range(k)]
    return names


def nodal_data(U, mesh, channels=None):
    """Concatenate the selected data channels -> (n_nodal_data(P), n_elem, Nn)."""
    parts = []
    for nm in _resolve_channels(channels):
        f = CHANNELS[nm][0](U, mesh)
        parts.append(f[None] if f.ndim == 2 else f)   # (1|width, n_elem, Nn)
    return jnp.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Features and model construction
# ---------------------------------------------------------------------------

def graph_features(U, mesh):
    """The graph-model input: (2, n_elem, Nn) = stacked
      [0] DG-FV density residual per NODE  (channel_residual, max-abs norm),
      [1] modal-energy spectrum per ELEMENT (modal_energy; axis 1 is the MODE
          index, kept ordered -- mode index = frequency).
    No one-hot: position awareness comes from xi and the DGSEM graph inside the
    model. The two rows share the (n_elem, Nn) shape because Nn modes = Nn
    nodes on a fixed-P mesh."""
    return jnp.stack([channel_residual(U, mesh), modal_energy(U, mesh)])


def alpha_features(U, mesh, model_type: str = "nodal", channels=None):
    """channels: list of channel names to build (defaults to the active
    NODAL_DATA_CHANNELS). Pass the checkpoint's own set (channels_from_meta) when
    evaluating a saved model so it sees the input it was trained with."""
    if model_type == "graph":
        return graph_features(U, mesh)
    if model_type == "nodal":
        return nodal_features(nodal_data(U, mesh, channels))   # [data..., one-hot]
    return modal_energy(U, mesh).T


def call_model(model, feats, mesh, model_type: str):
    """Apply a policy to PREBUILT features, hiding the signature split: graph
    models also take the mesh (reference-element constants stay out of the
    trainable pytree); legacy models take the features alone.

    Returns (alpha, alpha_boundary): alpha is the existing per-subcell-
    interface contract, unchanged for every model_type. alpha_boundary is the
    graph model's extra (n_elem,) blend for the true element-to-element
    interface flux (jax_dgsem.solver.hybrid_residual's alpha_boundary arg);
    None for "nodal"/"element", which have no such lever and fall back to the
    solver's old hardcoded entropy-stable-LF interface."""
    if model_type == "graph":
        return model(feats, mesh)
    return model(feats), None


def apply_alpha(model, U, mesh, model_type: str = "graph", channels=None):
    """State U -> raw (alpha, alpha_boundary): the single entry point every
    caller should use (features + model application in one step, any
    model_type)."""
    return call_model(model, alpha_features(U, mesh, model_type, channels),
                      mesh, model_type)


def build_alpha_model(model_type: str, P: int, width: int, kernel_size: int,
                      depth: int, key, alpha_init: float = 0.05,
                      stable_init: bool = True, n_data_channels: int = None,
                      b_res: bool = True, precondition: bool = False,
                      pno_channels: int = 8, pno_hidden: int = 32,
                      fusion_hidden: int = 32):
    if model_type == "graph":
        # P-independent by construction: no weight shape depends on P (`width`
        # is the GNN hidden size; kernel_size/b_res/precondition don't apply --
        # the graph model's preconditioning is formula-based, see the PNO).
        return GraphAlphaModel(pno_channels, pno_hidden, width, fusion_hidden,
                               depth, key=key, alpha_init=alpha_init,
                               stable_init=stable_init)
    if model_type == "nodal":
        nc = n_nodal_data(P) if n_data_channels is None else n_data_channels
        return NodalAlphaModel(P, width, kernel_size, depth, key=key,
                               alpha_init=alpha_init, stable_init=stable_init,
                               n_data_channels=nc, bool_res=b_res,
                               precondition=precondition)
    return AlphaModel(P + 1, width, kernel_size, depth, key=key,
                      alpha_init=alpha_init, stable_init=stable_init)


def model_meta(cfg) -> dict:
    """Serializable descriptor so loaders can rebuild the right template."""
    meta = {"model_type": cfg.model_type, "P": cfg.P, "width": cfg.width,
            "kernel_size": cfg.kernel_size, "depth": cfg.depth,
            "n_data_channels": n_nodal_data(cfg.P), "alpha_max": cfg.alpha_max,
            "precondition": cfg.precondition,
            "data_channels": list(NODAL_DATA_CHANNELS)}
    if cfg.model_type == "graph":
        meta.update(pno_channels=cfg.pno_channels, pno_hidden=cfg.pno_hidden,
                    fusion_hidden=cfg.fusion_hidden, precondition=False,
                    data_channels=["residual", "menergy"])
    return meta


def channels_from_meta(meta: dict):
    """The data-channel names a checkpoint was trained with. Prefers the list
    stored in the meta; for checkpoints saved before that list was recorded,
    falls back to the historical default inferred from the channel count
    (2 -> ["residual", "energy"], 1 -> ["residual"])."""
    if meta.get("data_channels"):
        return list(meta["data_channels"])
    n = int(meta.get("n_data_channels", 1))
    return {1: ["residual"], 2: ["residual", "energy"]}.get(
        n, list(NODAL_DATA_CHANNELS))


def build_from_meta(meta: dict, key):
    """Rebuild a model template from a model_meta.json dict. Uses the checkpoint's
    OWN n_data_channels so an old model loads even if the list changed since."""
    return build_alpha_model(meta.get("model_type", "element"), meta["P"],
                             meta["width"], meta["kernel_size"], meta["depth"],
                             key, n_data_channels=meta.get("n_data_channels"),
                             precondition=meta.get("precondition", False),
                             pno_channels=meta.get("pno_channels", 8),
                             pno_hidden=meta.get("pno_hidden", 32),
                             fusion_hidden=meta.get("fusion_hidden", 32))
