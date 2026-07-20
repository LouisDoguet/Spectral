"""Blending-factor machinery: modal features, Persson-Peraire indicator and
the clip/cap/diffuse post-processing, replicating lib/time/hybrid_solver.cpp
(computeAlpha) and lib/time/hybrid_alpha.h (modal_energy_features).
"""

import jax.numpy as jnp

from .solver import Mesh1D


def modal_energy(U, mesh: Mesh1D):
    """Normalized Legendre modal-energy spectrum of the density, per element.

    hybrid_alpha.h modal_energy_features: feat_j = m_j^2 / sum_k m_k^2.
    Returns (n_elem, Nn). This is the network input (element-local, so the
    policy is independent of the number of elements).
    """
    rho = U[0]                                   # (n_elem, Nn)
    m = jnp.einsum("kj,ej->ek", mesh.modal, rho)  # Legendre coefficients
    e = m * m
    total = jnp.sum(e, axis=1, keepdims=True)
    flat = jnp.zeros_like(e).at[:, 0].set(1.0)   # flat element -> [1,0,...,0]
    return jnp.where(total < 1e-30, flat, e / jnp.maximum(total, 1e-300))


def postprocess_alpha(raw, alpha_min: float = 0.001, alpha_max: float = 0.5,
                      diffuse: bool = True, hard_clip: bool = True):
    """hybrid_solver.cpp robustness tail, shared by the PP and the network
    branches: clip tiny/near-one values, cap at alpha_max, then one sweep of
    neighbour diffusion alpha_e = max(alpha_e, 0.5 * max(neighbours)).

    hard_clip=False skips the step-function clipping (zero gradient regions)
    for training; keep it True at deployment to match the C++ solver.
    """
    a = raw
    if hard_clip:
        a = jnp.where(a < alpha_min, 0.0, a)
        a = jnp.where(a > 1.0 - alpha_min, 1.0, a)
    a = jnp.minimum(a, alpha_max)
    # Neighbour diffusion is only defined for the per-element (1D) policy; the
    # nodal policy (2D, (n_elem, P)) skips it.
    if diffuse and a.ndim == 1 and a.shape[0] > 1:
        left = jnp.concatenate([a[1:2] * 0.0, a[:-1]])   # neighbour to the left
        right = jnp.concatenate([a[1:], a[-2:-1] * 0.0])  # neighbour to the right
        a = jnp.maximum(a, 0.5 * jnp.maximum(left, right))
    return a


def persson_peraire_indicator(U, mesh: Mesh1D):
    """The raw Persson-Peraire modal-energy indicator E_ind, per element
    (n_elem,). This is the single scalar PP thresholds: the fraction of the
    density modal energy sitting in the top mode(s). Small in smooth elements,
    O(0.1) at a discontinuity. Used as a network input channel."""
    P = mesh.P
    m = jnp.einsum("kj,ej->ek", mesh.modal, U[0])
    e = m * m
    total = jnp.sum(e, axis=1)
    total_m1 = total - e[:, P]
    E_N = jnp.where(total > 1e-300, e[:, P] / jnp.maximum(total, 1e-300), 0.0)
    E_Nm = jnp.where((P > 0) & (total_m1 > 1e-300),
                     e[:, P - 1] / jnp.maximum(total_m1, 1e-300), 0.0)
    return jnp.maximum(E_N, E_Nm)


def persson_peraire_alpha(U, mesh: Mesh1D, alpha_min: float = 0.001,
                          alpha_max: float = 0.5, diffuse: bool = True):
    """Persson-Peraire modal energy indicator (hybrid_solver.cpp Eqs. 40-48).

    This is the built-in (non-learned) alpha used to generate reference
    trajectories and as the baseline the network competes against.
    """
    P = mesh.P
    N1 = float(P + 1)
    T = 0.5 * 10.0 ** (-1.8 * N1 ** 0.25)
    s = 9.21024

    rho = U[0]
    m = jnp.einsum("kj,ej->ek", mesh.modal, rho)
    e = m * m
    total = jnp.sum(e, axis=1)

    total_m1 = total - e[:, P]
    E_N = e[:, P] / total
    E_Nm = jnp.where((P > 0) & (total_m1 > 1e-300), e[:, P - 1] / jnp.maximum(total_m1, 1e-300), 0.0)
    Eind = jnp.maximum(E_N, E_Nm)

    raw = 1.0 / (1.0 + jnp.exp(-s / T * (Eind - T)))
    raw = jnp.where(total < 1e-300, 0.0, raw)

    return postprocess_alpha(raw, alpha_min, alpha_max, diffuse, hard_clip=True)
