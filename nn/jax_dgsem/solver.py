"""Entropy-stable hybrid DGSEM residual + RK4, replicating lib/space/mesh.cpp
(computeHybridResidual path) and lib/time/hybrid_solver.cpp.

State U: (3, n_elem, Nn) with Nn = P+1 GLL nodes per element.
Residual convention matches C++: divF holds +dF/dx and dU/dt = -divF, so the
functions below return rhs = -divF directly.

Everything is pure jnp -> the whole time loop is differentiable and the alpha
input (per-element, shape (n_elem,)) carries gradients back to the network.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from .basis import GLLBasis
from .physics import physical_flux, chandrashekar_ec, entropy_stable_lf


class Mesh1D(eqx.Module):
    """Static mesh description (uniform elements on [xL, xR]).

    bc_left/bc_right: ("wall", U_ghost) fixed Dirichlet ghost state,
    ("reflective", None) mirror with reversed momentum — the two types in
    lib/boundary_conditions — or ("periodic", None) on BOTH sides, which
    couples the last element's right face to the first element's left face
    (the setup used for the MUSCL-reference training).
    """

    quads: jnp.ndarray
    weights: jnp.ndarray
    D: jnp.ndarray
    modal: jnp.ndarray
    n_elem: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    dx: float
    inv_J: float
    bc_left_kind: str = eqx.field(static=True)
    bc_right_kind: str = eqx.field(static=True)
    bc_left_state: jnp.ndarray
    bc_right_state: jnp.ndarray

    def __init__(self, basis: GLLBasis, n_elem: int, xL: float, xR: float,
                 bc_left=("reflective", None), bc_right=("reflective", None)):
        self.quads = basis.quads
        self.weights = basis.weights
        self.D = basis.D
        self.modal = basis.modal
        self.n_elem = n_elem
        self.P = basis.P
        self.dx = (xR - xL) / n_elem
        self.inv_J = 2.0 / self.dx          # element.cpp setJ: J = dx/2
        self.bc_left_kind = bc_left[0]
        self.bc_right_kind = bc_right[0]
        if (self.bc_left_kind == "periodic") != (self.bc_right_kind == "periodic"):
            raise ValueError("periodic BC must be set on BOTH sides")
        zero = jnp.zeros(3)
        self.bc_left_state = zero if bc_left[1] is None else jnp.asarray(bc_left[1])
        self.bc_right_state = zero if bc_right[1] is None else jnp.asarray(bc_right[1])

    @property
    def periodic(self) -> bool:
        return self.bc_left_kind == "periodic"

    def node_positions(self, xL: float = 0.0) -> jnp.ndarray:
        """Physical x of every node, shape (n_elem, Nn)."""
        elem_left = xL + self.dx * jnp.arange(self.n_elem)
        return elem_left[:, None] + (self.quads[None, :] + 1.0) * self.dx / 2.0


def _ghost_state(kind: str, fixed: jnp.ndarray, U_int: jnp.ndarray):
    """boundary_conditions.cpp: Wall = fixed Dirichlet state, Reflective =
    mirror with reversed momentum."""
    if kind == "wall":
        return fixed
    if kind == "reflective":
        return jnp.stack([U_int[0], -U_int[1], U_int[2]])
    raise ValueError(f"unknown BC kind '{kind}'")


# ---------------------------------------------------------------------------
# Subcell interface fluxes B_0..B_Nn per element (mesh.cpp)
# ---------------------------------------------------------------------------

def _subcell_flux_dg(U, mesh: Mesh1D):
    """buildSubcellFluxDG: SBP telescoping of the split-form EC volume.

    B_0 = F(u_0); B_{j+1} = B_j + 2 w_j sum_k D_jk f*_EC(u_j, u_k);
    B_Nn overwritten with F(u_{Nn-1}). Returns (3, n_elem, Nn+1).
    """
    # EC flux for every node pair of every element: (3, n_elem, Nn, Nn)
    UL = U[:, :, :, None]
    UR = U[:, :, None, :]
    ec = chandrashekar_ec(UL, UR)

    # d_j = sum_k D_jk ec_jk ; increment 2 w_j d_j
    d = jnp.einsum("jk,cejk->cej", mesh.D, ec)
    inc = 2.0 * mesh.weights[None, None, :] * d

    B0 = physical_flux(U[:, :, 0])                       # (3, n_elem)
    Bend = physical_flux(U[:, :, -1])
    # interior B_1..B_{Nn-1} from the first Nn-1 increments
    B_interior = B0[:, :, None] + jnp.cumsum(inc[:, :, :-1], axis=2)
    return jnp.concatenate(
        [B0[:, :, None], B_interior, Bend[:, :, None]], axis=2)


def _subcell_flux_fv(U, mesh: Mesh1D):
    """buildSubcellFluxFV: first-order ES flux across each subcell interface."""
    B0 = physical_flux(U[:, :, 0])
    Bend = physical_flux(U[:, :, -1])
    B_interior = entropy_stable_lf(U[:, :, :-1], U[:, :, 1:])  # (3, n_elem, Nn-1)
    return jnp.concatenate(
        [B0[:, :, None], B_interior, Bend[:, :, None]], axis=2)


# ---------------------------------------------------------------------------
# Shared surface term + closure
# ---------------------------------------------------------------------------

def _apply_surface_and_mass_inverse(divF, U, mesh: Mesh1D):
    """applyEntropyStableInterfaces + applyEntropyStableBoundaries +
    applyMassInverse, in the exact C++ order."""
    # Interior interfaces: L_i = (F* - F_int) lifts
    UL = U[:, :-1, -1]                    # (3, n_elem-1) left of interface
    UR = U[:, 1:, 0]
    fstar = entropy_stable_lf(UL, UR)
    divF = divF.at[:, :-1, -1].add(fstar - physical_flux(UL))
    divF = divF.at[:, 1:, 0].add(physical_flux(UR) - fstar)

    if mesh.periodic:
        # Wrap-around interface: last element's right face <-> first element's
        # left face, treated exactly like an interior interface. Entropy
        # stable and freestream preserving (constant U -> zero lift).
        ULp = U[:, -1, -1]
        URp = U[:, 0, 0]
        fp = entropy_stable_lf(ULp, URp)
        divF = divF.at[:, -1, -1].add(fp - physical_flux(ULp))
        divF = divF.at[:, 0, 0].add(physical_flux(URp) - fp)
    else:
        # Left domain boundary (elem 0, node 0): lift (F_int - F*)
        Ui = U[:, 0, 0]
        Ug = _ghost_state(mesh.bc_left_kind, mesh.bc_left_state, Ui)
        fs = entropy_stable_lf(Ug, Ui)
        divF = divF.at[:, 0, 0].add(physical_flux(Ui) - fs)

        # Right domain boundary (last elem, node Nn-1): lift (F* - F_int)
        Ui = U[:, -1, -1]
        Ug = _ghost_state(mesh.bc_right_kind, mesh.bc_right_state, Ui)
        fs = entropy_stable_lf(Ui, Ug)
        divF = divF.at[:, -1, -1].add(fs - physical_flux(Ui))

    # applyMassInverse: divF <- (1/J) Minv divF with M = diag(w)
    return divF * mesh.inv_J / mesh.weights[None, None, :]


def dg_residual(U, mesh: Mesh1D):
    """computeDGResidual, returned as divF (before the sign flip)."""
    B = _subcell_flux_dg(U, mesh)
    divF = B[:, :, 1:] - B[:, :, :-1]     # storeSubcellDivergence
    return _apply_surface_and_mass_inverse(divF, U, mesh)


def fv_residual(U, mesh: Mesh1D):
    """computeFVResidual, returned as divF."""
    B = _subcell_flux_fv(U, mesh)
    divF = B[:, :, 1:] - B[:, :, :-1]
    return _apply_surface_and_mass_inverse(divF, U, mesh)


def hybrid_residual(U, alpha, mesh: Mesh1D):
    """computeHybridResidual: R = (1-alpha) R_DG + alpha R_FV, then the RK4
    convention rhs = -divF. alpha is per element, shape (n_elem,)."""
    a = alpha[None, :, None]
    divF = (1.0 - a) * dg_residual(U, mesh) + a * fv_residual(U, mesh)
    return -divF


# ---------------------------------------------------------------------------
# Time integration (hybrid_solver.cpp: alpha frozen across the 4 RK4 stages)
# ---------------------------------------------------------------------------

def rk4_step(U, alpha, dt, mesh: Mesh1D):
    """Classical RK4 with the residual evaluated at frozen alpha."""
    k1 = hybrid_residual(U, alpha, mesh)
    k2 = hybrid_residual(U + 0.5 * dt * k1, alpha, mesh)
    k3 = hybrid_residual(U + 0.5 * dt * k2, alpha, mesh)
    k4 = hybrid_residual(U + dt * k3, alpha, mesh)
    return U + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def time_loop(U0, alpha_fn, n_steps: int, dt, mesh: Mesh1D,
              return_trajectory: bool = True):
    """Advance n_steps. alpha_fn(U) -> (n_elem,) is called ONCE per step
    (HybridDGSEM::step computes alpha before the stages), so the network cost
    matches the C++ deployment exactly and gradients flow through alpha_fn.

    Returns (U_final, trajectory) with trajectory (n_steps, 3, n_elem, Nn)
    holding the state AFTER each step, or (U_final, None).
    """

    def body(U, _):
        alpha = alpha_fn(U)
        U_next = rk4_step(U, alpha, dt, mesh)
        return U_next, (U_next if return_trajectory else None)

    U_final, traj = jax.lax.scan(body, U0, None, length=n_steps)
    return U_final, traj


def cfl_time_step(U, mesh: Mesh1D, cfl: float):
    """mesh.cpp computeCFLTimeStep: cfl * dx / ((2P+1) * lambda_max)."""
    from .physics import max_wave_speed
    lam = jnp.max(max_wave_speed(U))
    return cfl * mesh.dx / ((2.0 * mesh.P + 1.0) * lam)
