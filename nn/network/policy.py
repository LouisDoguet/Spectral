"""Routing between the two alpha policies (per-element and nodal/subcell), and
the definition of the NODAL NETWORK INPUT.

  "element": legacy per-element policy. Input = normalized modal-energy
             spectrum (P+1 channels) over the elements; output alpha (n_elem,).
  "nodal"  : per interior subcell interface. Input = a stack of per-node DATA
             channels followed by the one-hot position block; output alpha
             (n_elem, P).

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
from network.model import AlphaModel, NodalAlphaModel, nodal_features


# ---------------------------------------------------------------------------
# Nodal data channels  (edit this list to change the input)
# ---------------------------------------------------------------------------

def channel_residual(U, mesh):
    """DG - FV density residual per node, normalized to ~[-1, 1]. The per-NODE
    troubled signal: where the high-order and robust operators disagree. This
    is what lets the network localize at the subcell level (beyond PP)."""
    res = (dg_residual(U, mesh) - fv_residual(U, mesh))[0]      # (n_elem, Nn)
    return res / (jnp.max(jnp.abs(res)) + 1e-8)


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

def alpha_features(U, mesh, model_type: str = "nodal", channels=None):
    """channels: list of channel names to build (defaults to the active
    NODAL_DATA_CHANNELS). Pass the checkpoint's own set (channels_from_meta) when
    evaluating a saved model so it sees the input it was trained with."""
    if model_type == "nodal":
        return nodal_features(nodal_data(U, mesh, channels))   # [data..., one-hot]
    return modal_energy(U, mesh).T


def build_alpha_model(model_type: str, P: int, width: int, kernel_size: int,
                      depth: int, key, alpha_init: float = 0.05,
                      stable_init: bool = True, n_data_channels: int = None,
                      b_res: bool = True, precondition: bool = False):
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
    return {"model_type": cfg.model_type, "P": cfg.P, "width": cfg.width,
            "kernel_size": cfg.kernel_size, "depth": cfg.depth,
            "n_data_channels": n_nodal_data(cfg.P), "alpha_max": cfg.alpha_max,
            "precondition": cfg.precondition,
            "data_channels": list(NODAL_DATA_CHANNELS)}


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
                             precondition=meta.get("precondition", False))
