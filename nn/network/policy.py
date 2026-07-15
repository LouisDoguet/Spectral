"""Routing between the two alpha policies (per-element and nodal/subcell).

Keeps the model_type switch in ONE place so train / main / viz / compare all
build the same features and model for a given config.

  "element": legacy per-element policy. Input = normalized modal-energy
             spectrum (P+1 channels) over the elements; output alpha (n_elem,).
  "nodal"  : per interior subcell interface. Input = [density, one-hot node
             position] over the node sequence; output alpha (n_elem, P).
"""

from jax_dgsem.indicator import modal_energy
from network.model import AlphaModel, NodalAlphaModel, nodal_features


def alpha_features(U, mesh, model_type: str = "nodal"):
    if model_type == "nodal":
        return nodal_features(U, mesh)
    return modal_energy(U, mesh).T


def build_alpha_model(model_type: str, P: int, width: int, kernel_size: int,
                      depth: int, key, alpha_init: float = 0.05):
    if model_type == "nodal":
        return NodalAlphaModel(P, width, kernel_size, depth, key=key,
                               alpha_init=alpha_init)
    return AlphaModel(P + 1, width, kernel_size, depth, key=key,
                      alpha_init=alpha_init)


def model_meta(cfg) -> dict:
    """Serializable descriptor so loaders can rebuild the right template."""
    return {"model_type": cfg.model_type, "P": cfg.P, "width": cfg.width,
            "kernel_size": cfg.kernel_size, "depth": cfg.depth,
            "in_channels": cfg.P + 1, "alpha_max": cfg.alpha_max}


def build_from_meta(meta: dict, key):
    """Rebuild a model template from a model_meta.json dict."""
    return build_alpha_model(meta.get("model_type", "element"), meta["P"],
                             meta["width"], meta["kernel_size"], meta["depth"],
                             key)
