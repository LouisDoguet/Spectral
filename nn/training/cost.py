"""Cost function (Bois et al. Sect. 3.2) adapted to the alpha control.

C(U, U_ref) = w_osc * C_osc + w_acc * C_acc + w_alpha * C_alpha
  C_osc  : L1 error of the discrete second derivative on a fine uniform grid
  C_acc  : L1 error on the fine uniform grid
  C_alpha: L2 penalization of the control (replaces the paper's C_vis; pushes
           alpha -> 0, i.e. toward the pure high-order DG scheme)

Both the coarse DG solution and the (finer-mesh) reference are projected onto
the SAME uniform grid before comparison, so meshes of different resolutions
are comparable. For the Euler system costs are averaged over the three
conservative variables (paper: "we simply take the average cost").
"""

import numpy as np
import jax.numpy as jnp

from jax_dgsem.basis import lagrange_interpolation_matrix, gll_nodes


def uniform_projector(P: int, n_elem: int, pts_per_elem: int):
    """Sampling of the piecewise-polynomial DG solution on a uniform grid.

    Points are the centers of pts_per_elem uniform cells per element (the
    paper's Pi_ref projection onto the reference FV grid). Returns
    (project, dx_ref) with project: (3, n_elem, Nn) -> (3, n_elem*pts_per_elem).
    """
    quads = gll_nodes(P)
    j = np.arange(pts_per_elem)
    xi = -1.0 + (2.0 * j + 1.0) / pts_per_elem
    T = jnp.asarray(lagrange_interpolation_matrix(quads, xi))  # (K, Nn)

    def project(U):
        vals = jnp.einsum("qj,cej->ceq", T, U)
        return vals.reshape(U.shape[0], -1)

    return project


def _dxx(v, dx_ref):
    """Periodic finite-difference second derivative on the uniform grid.

    The training problem is periodic, so the stencil wraps (jnp.roll) instead
    of dropping the endpoints; this keeps the oscillation penalty meaningful
    across the wrap-around interface too."""
    return (jnp.roll(v, 1, axis=1) - 2.0 * v + jnp.roll(v, -1, axis=1)) \
        / (dx_ref * dx_ref)


def cost_step(U_proj, ref_proj, alpha, dx_ref,
              w_osc: float, w_acc: float, w_alpha: float):
    """Local-in-time cost C(U^n, U_ref^n) on already-projected states."""
    c_acc = dx_ref * jnp.sum(jnp.abs(U_proj - ref_proj), axis=1).mean()
    c_osc = dx_ref * jnp.sum(
        jnp.abs(_dxx(U_proj, dx_ref) - _dxx(ref_proj, dx_ref)), axis=1).mean()
    c_alpha = jnp.sum(alpha * alpha)
    return w_osc * c_osc + w_acc * c_acc + w_alpha * c_alpha


def cost_terms(U_proj, ref_proj, alpha, dx_ref):
    """Unweighted (C_osc, C_acc, C_alpha) for monitoring."""
    c_acc = dx_ref * jnp.sum(jnp.abs(U_proj - ref_proj), axis=1).mean()
    c_osc = dx_ref * jnp.sum(
        jnp.abs(_dxx(U_proj, dx_ref) - _dxx(ref_proj, dx_ref)), axis=1).mean()
    c_alpha = jnp.sum(alpha * alpha)
    return c_osc, c_acc, c_alpha
