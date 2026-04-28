"""Exact solution for the Sod shock tube (1D Euler equations).

Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", Ch. 4.
"""
import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4


def _f(p_star, rho, p, c):
    """Pressure function for the exact Riemann solver."""
    if p_star > p:  # shock
        A = 2.0 / ((GAMMA + 1) * rho)
        B = (GAMMA - 1) / (GAMMA + 1) * p
        return (p_star - p) * np.sqrt(A / (p_star + B))
    else:           # rarefaction
        return (2.0 * c / (GAMMA - 1)) * ((p_star / p) ** ((GAMMA - 1) / (2 * GAMMA)) - 1.0)


def sod_exact(x, t,
              rho_L=1.0, u_L=0.0, p_L=1.0,
              rho_R=0.125, u_R=0.0, p_R=0.1,
              x0=0.5):
    """
    Exact Sod shock tube solution at positions x and time t.

    Returns
    -------
    rho, u, p : ndarrays of the same shape as x
    """
    x = np.asarray(x, dtype=float)

    if t == 0.0:
        rho = np.where(x < x0, rho_L, rho_R)
        u   = np.where(x < x0, u_L,   u_R)
        p   = np.where(x < x0, p_L,   p_R)
        return rho, u, p

    c_L = np.sqrt(GAMMA * p_L / rho_L)
    c_R = np.sqrt(GAMMA * p_R / rho_R)

    # Star-region pressure (root finding)
    p_star = brentq(
        lambda ps: _f(ps, rho_L, p_L, c_L) + _f(ps, rho_R, p_R, c_R) + (u_R - u_L),
        1e-12, max(p_L, p_R) * 100.0,
    )

    # Star-region velocity
    u_star = u_L - _f(p_star, rho_L, p_L, c_L)

    # Star-region densities
    rho_Lstar = rho_L * (p_star / p_L) ** (1.0 / GAMMA)           # isentropic (rarefaction)
    c_Lstar   = c_L   * (p_star / p_L) ** ((GAMMA - 1) / (2 * GAMMA))
    mu2       = (GAMMA - 1) / (GAMMA + 1)
    rho_Rstar = rho_R * (p_star / p_R + mu2) / (mu2 * p_star / p_R + 1.0)  # Rankine-Hugoniot

    # Wave speeds
    S_HL = u_L - c_L       # head of left rarefaction
    S_TL = u_star - c_Lstar # tail of left rarefaction
    S_C  = u_star           # contact discontinuity
    S_R  = u_R + c_R * np.sqrt((GAMMA + 1) / (2 * GAMMA) * p_star / p_R
                                + (GAMMA - 1) / (2 * GAMMA))  # right shock

    xi = (x - x0) / t  # self-similar coordinate

    rho_out = np.empty_like(x)
    u_out   = np.empty_like(x)
    p_out   = np.empty_like(x)

    # Left undisturbed state
    m = xi <= S_HL
    rho_out[m], u_out[m], p_out[m] = rho_L, u_L, p_L

    # Inside rarefaction fan
    m = (xi > S_HL) & (xi <= S_TL)
    u_fan        = 2.0 / (GAMMA + 1) * (c_L + xi[m])
    c_fan        = c_L - (GAMMA - 1) / 2.0 * u_fan
    rho_out[m]   = rho_L * (c_fan / c_L) ** (2.0 / (GAMMA - 1))
    u_out[m]     = u_fan
    p_out[m]     = p_L   * (c_fan / c_L) ** (2.0 * GAMMA / (GAMMA - 1))

    # Left star state
    m = (xi > S_TL) & (xi <= S_C)
    rho_out[m], u_out[m], p_out[m] = rho_Lstar, u_star, p_star

    # Right star state
    m = (xi > S_C) & (xi <= S_R)
    rho_out[m], u_out[m], p_out[m] = rho_Rstar, u_star, p_star

    # Right undisturbed state
    m = xi > S_R
    rho_out[m], u_out[m], p_out[m] = rho_R, u_R, p_R

    return rho_out, u_out, p_out
