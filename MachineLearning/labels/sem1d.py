"""Lightweight Python 1D SEM forward model for offline label generation.

Mirrors the C++ solver (RK4 + Rusanov + Dirichlet BCs + diffusion) exactly
so that the optimal-eps search produces consistent labels.
Not optimised for speed — used offline only.
"""
import numpy as np
from data.gll import nodes_weights, deriv_matrix

GAMMA = 1.4


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def make_geometry(n_elem: int, P: int, xL: float = 0.0, xR: float = 1.0) -> dict:
    """Pre-compute all mesh geometry needed by the solver."""
    xi, w = nodes_weights(P)
    D     = deriv_matrix(xi, P)
    n     = P + 1
    dx    = (xR - xL) / n_elem
    J     = dx / 2.0
    invJ  = 2.0 / dx

    x = np.array([
        xL + e * dx + (xi[q] + 1.0) * dx / 2.0
        for e in range(n_elem) for q in range(n)
    ])

    return {
        "xi": xi, "w": w, "D": D,
        "J": J, "invJ": invJ,
        "n": n, "n_elem": n_elem, "n_total": n_elem * n,
        "x": x, "xL": xL, "xR": xR, "dx": dx,
    }


# ---------------------------------------------------------------------------
# Physics (mirrors physics.cpp)
# ---------------------------------------------------------------------------

def _flux(rho, rhou, enrg):
    """Euler fluxes (F1, F2, F3) — works for scalars or arrays."""
    u = rhou / rho
    p = (GAMMA - 1) * (enrg - 0.5 * rhou ** 2 / rho)
    return rhou, rhou * u + p, u * (enrg + p)


def _wave_speed(rho, rhou, enrg):
    """Local maximum wave speed |u| + c."""
    p = (GAMMA - 1) * (enrg - 0.5 * rhou ** 2 / rho)
    return abs(rhou / rho) + np.sqrt(GAMMA * p / rho)


def _rusanov(fL, fR, uL, uR, lam):
    """Rusanov (LLF) numerical flux — mirrors reimann::Rusanov in physics.cpp."""
    return 0.5 * (fL + fR) - 0.5 * lam * (uR - uL)


# ---------------------------------------------------------------------------
# Residual  R(U) = dU/dt = -dF/dx + diffusion
# ---------------------------------------------------------------------------

def compute_residual(rho, rhou, enrg, eps, geom, bc_L, bc_R):
    """
    Compute dU/dt for the full mesh.

    Parameters
    ----------
    rho, rhou, enrg : (n_total,) state arrays
    eps             : (n_total,) diffusion coefficient per node
    geom            : dict from make_geometry
    bc_L, bc_R      : (rho, rhou, e) tuples for Dirichlet boundaries

    Returns
    -------
    (drho_dt, drhou_dt, denrg_dt) : each (n_total,)
    """
    D, invJ   = geom["D"], geom["invJ"]
    n, n_elem = geom["n"], geom["n_elem"]
    w, P      = geom["w"], geom["n"] - 1

    dF1 = np.zeros(geom["n_total"])
    dF2 = np.zeros(geom["n_total"])
    dF3 = np.zeros(geom["n_total"])

    # --- Element-interior flux divergence: invJ * D * F ---
    for e in range(n_elem):
        sl = slice(e * n, (e + 1) * n)
        F1, F2, F3 = _flux(rho[sl], rhou[sl], enrg[sl])
        dF1[sl] = invJ * (D @ F1)
        dF2[sl] = invJ * (D @ F2)
        dF3[sl] = invJ * (D @ F3)

    # --- Interior interface Riemann corrections ---
    for e in range(n_elem - 1):
        iL = e * n + P          # right node of left element
        iR = (e + 1) * n        # left  node of right element

        UL = np.array([rho[iL],  rhou[iL],  enrg[iL]])
        UR = np.array([rho[iR],  rhou[iR],  enrg[iR]])
        FL = np.array(_flux(*UL))
        FR = np.array(_flux(*UR))
        lam = max(_wave_speed(*UL), _wave_speed(*UR))
        Fs  = _rusanov(FL, FR, UL, UR, lam)

        invWJ_L = invJ / w[P]
        invWJ_R = invJ / w[0]

        # Right face of left element:  correction = invWJ * (f* - f_local)
        dF1[iL] += invWJ_L * (Fs[0] - FL[0])
        dF2[iL] += invWJ_L * (Fs[1] - FL[1])
        dF3[iL] += invWJ_L * (Fs[2] - FL[2])

        # Left face of right element: correction = invWJ * (f_local - f*)
        dF1[iR] += invWJ_R * (FR[0] - Fs[0])
        dF2[iR] += invWJ_R * (FR[1] - Fs[1])
        dF3[iR] += invWJ_R * (FR[2] - Fs[2])

    # --- Dirichlet boundary conditions (mirrors mesh.cpp applyDirichlet) ---
    # Left boundary
    u1_L, u2_L, u3_L   = bc_L
    u1_int, u2_int, u3_int = rho[0], rhou[0], enrg[0]
    f_ext = np.array(_flux(u1_L,   u2_L,   u3_L))
    f_int = np.array(_flux(u1_int, u2_int, u3_int))
    lam_L = max(_wave_speed(u1_L, u2_L, u3_L), _wave_speed(u1_int, u2_int, u3_int))
    Fs_L  = _rusanov(f_ext, f_int,
                     np.array([u1_L, u2_L, u3_L]),
                     np.array([u1_int, u2_int, u3_int]), lam_L)
    invWJ_L = invJ / w[0]
    # Left face of first element: correction = invWJ * (f_int - f*)
    dF1[0] += invWJ_L * (f_int[0] - Fs_L[0])
    dF2[0] += invWJ_L * (f_int[1] - Fs_L[1])
    dF3[0] += invWJ_L * (f_int[2] - Fs_L[2])

    # Right boundary
    u1_R, u2_R, u3_R   = bc_R
    last = (n_elem - 1) * n + P
    u1_int, u2_int, u3_int = rho[last], rhou[last], enrg[last]
    f_ext = np.array(_flux(u1_R,   u2_R,   u3_R))
    f_int = np.array(_flux(u1_int, u2_int, u3_int))
    lam_R = max(_wave_speed(u1_R, u2_R, u3_R), _wave_speed(u1_int, u2_int, u3_int))
    Fs_R  = _rusanov(f_int, f_ext,
                     np.array([u1_int, u2_int, u3_int]),
                     np.array([u1_R, u2_R, u3_R]), lam_R)
    invWJ_R = invJ / w[P]
    # Right face of last element: correction = invWJ * (f* - f_int)
    dF1[last] += invWJ_R * (Fs_R[0] - f_int[0])
    dF2[last] += invWJ_R * (Fs_R[1] - f_int[1])
    dF3[last] += invWJ_R * (Fs_R[2] - f_int[2])

    # --- Diffusion: subtract eps * d²U/dx² from dF (mirrors diffusion.cpp) ---
    for e in range(n_elem):
        sl    = slice(e * n, (e + 1) * n)
        eps_e = eps[sl]

        du1 = invJ * (D @ rho[sl]);   du1 *= eps_e;  dF1[sl] -= invJ * (D @ du1)
        du2 = invJ * (D @ rhou[sl]);  du2 *= eps_e;  dF2[sl] -= invJ * (D @ du2)
        du3 = invJ * (D @ enrg[sl]);  du3 *= eps_e;  dF3[sl] -= invJ * (D @ du3)

    # dU/dt = -dF/dx + diffusion  (diffusion already subtracted from dF)
    return -dF1, -dF2, -dF3


# ---------------------------------------------------------------------------
# RK4 time step (mirrors rk4.cpp RK4::step)
# ---------------------------------------------------------------------------

def step(rho, rhou, enrg, eps, dt, geom, bc_L, bc_R):
    """
    Advance (rho, rhou, enrg) by one RK4 step of size dt.

    Returns
    -------
    (rho_new, rhou_new, enrg_new) : each (n_total,)
    """
    def R(u1, u2, u3):
        return compute_residual(u1, u2, u3, eps, geom, bc_L, bc_R)

    k1 = R(rho, rhou, enrg)
    k2 = R(rho  + 0.5 * dt * k1[0], rhou + 0.5 * dt * k1[1], enrg + 0.5 * dt * k1[2])
    k3 = R(rho  + 0.5 * dt * k2[0], rhou + 0.5 * dt * k2[1], enrg + 0.5 * dt * k2[2])
    k4 = R(rho  +       dt * k3[0], rhou +       dt * k3[1], enrg +       dt * k3[2])

    c = dt / 6.0
    return (
        rho  + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        rhou + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        enrg + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    )
