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


# The nodal input's data channels, in order. Comment out a line to remove that
# channel (e.g. remove "energy" to go back to residual-only).
NODAL_DATA_CHANNELS = [
    ("residual", channel_residual),
    ("energy",   channel_energy),
]


def n_nodal_data() -> int:
    return len(NODAL_DATA_CHANNELS)


def nodal_data(U, mesh):
    """Stack the enabled data channels: (len(NODAL_DATA_CHANNELS), n_elem, Nn)."""
    return jnp.stack([fn(U, mesh) for _, fn in NODAL_DATA_CHANNELS])


# ---------------------------------------------------------------------------
# Features and model construction
# ---------------------------------------------------------------------------

def alpha_features(U, mesh, model_type: str = "nodal"):
    if model_type == "nodal":
        return nodal_features(nodal_data(U, mesh))     # [data..., one-hot]
    return modal_energy(U, mesh).T


def build_alpha_model(model_type: str, P: int, width: int, kernel_size: int,
                      depth: int, key, alpha_init: float = 0.05,
                      stable_init: bool = True, n_data_channels: int = None):
    if model_type == "nodal":
        nc = n_nodal_data() if n_data_channels is None else n_data_channels
        return NodalAlphaModel(P, width, kernel_size, depth, key=key,
                               alpha_init=alpha_init, stable_init=stable_init,
                               n_data_channels=nc)
    return AlphaModel(P + 1, width, kernel_size, depth, key=key,
                      alpha_init=alpha_init, stable_init=stable_init)


def model_meta(cfg) -> dict:
    """Serializable descriptor so loaders can rebuild the right template."""
    return {"model_type": cfg.model_type, "P": cfg.P, "width": cfg.width,
            "kernel_size": cfg.kernel_size, "depth": cfg.depth,
            "n_data_channels": n_nodal_data(), "alpha_max": cfg.alpha_max}


def build_from_meta(meta: dict, key):
    """Rebuild a model template from a model_meta.json dict. Uses the checkpoint's
    OWN n_data_channels so an old model loads even if the list changed since."""
    return build_alpha_model(meta.get("model_type", "element"), meta["P"],
                             meta["width"], meta["kernel_size"], meta["depth"],
                             key, n_data_channels=meta.get("n_data_channels"))
