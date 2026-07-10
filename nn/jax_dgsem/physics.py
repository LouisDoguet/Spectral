"""1D Euler physics + entropy fluxes, replicating lib/phy/*.cpp.

State convention throughout the package: U has shape (3, ...) with
U[0]=rho, U[1]=rho*u, U[2]=E. All functions broadcast over trailing dims.
"""

import jax.numpy as jnp

GAMMA = 1.4
GM1 = GAMMA - 1.0


def pressure(U):
    """p = (gamma-1)(E - 0.5 rho u^2) (physics.cpp getP)."""
    rho, rhou, E = U[0], U[1], U[2]
    return GM1 * (E - 0.5 * rhou * rhou / rho)


def physical_flux(U):
    """Euler flux F(U) (mesh.cpp physicalFlux)."""
    rho, rhou, E = U[0], U[1], U[2]
    u = rhou / rho
    p = GM1 * (E - 0.5 * rhou * u)
    return jnp.stack([rhou, rhou * u + p, u * (E + p)])


def max_wave_speed(U):
    """|u| + c (physics.cpp computeMaxWaveSpeed)."""
    rho, rhou = U[0], U[1]
    u = rhou / rho
    p = pressure(U)
    return jnp.abs(u) + jnp.sqrt(GAMMA * p / rho)


def logmean(a, b):
    """Ismail-Roe stable logarithmic mean (entropy_flux.cpp logmean).

    Branchless jnp.where version. With f = (a-b)/(a+b) one has
    (b-a)/(log b - log a) = (a+b) / (2 * artanh(f)/f), and artanh(f)/f is the
    C++ Taylor series 1 + f2/3 + f4/5 + f6/7 for small f. The unused branch is
    fed a safe f so its NaN/Inf never pollutes gradients.
    """
    f = (a - b) / (a + b)
    f2 = f * f
    small = f2 < 1e-4
    taylor = 1.0 + f2 * (1.0 / 3.0 + f2 * (1.0 / 5.0 + f2 / 7.0))
    f_safe = jnp.where(small, 0.5, f)
    exact = jnp.arctanh(f_safe) / f_safe
    G = jnp.where(small, taylor, exact)
    return (a + b) / (2.0 * G)


def chandrashekar_ec(UL, UR):
    """Entropy-conservative Chandrashekar two-point flux (entropy_flux.cpp)."""
    rho1, rhou1, e1 = UL[0], UL[1], UL[2]
    rho2, rhou2, e2 = UR[0], UR[1], UR[2]
    u1, u2 = rhou1 / rho1, rhou2 / rho2
    p1 = GM1 * (e1 - 0.5 * rhou1 * u1)
    p2 = GM1 * (e2 - 0.5 * rhou2 * u2)

    beta1 = 0.5 * rho1 / p1
    beta2 = 0.5 * rho2 / p2

    rho_ln = logmean(rho1, rho2)
    beta_ln = logmean(beta1, beta2)

    u_avg = 0.5 * (u1 + u2)
    rho_avg = 0.5 * (rho1 + rho2)
    beta_avg = 0.5 * (beta1 + beta2)

    p_hat = 0.5 * rho_avg / beta_avg
    h_hat = 1.0 / (2.0 * GM1 * beta_ln) + p_hat / rho_ln + 0.5 * u_avg * u_avg

    f1 = rho_ln * u_avg
    return jnp.stack([f1, f1 * u_avg + p_hat, f1 * h_hat])


def entropy_stable_lf(UL, UR):
    """EC flux + local Lax-Friedrichs dissipation (entropy_flux.cpp)."""
    f = chandrashekar_ec(UL, UR)
    lam = jnp.maximum(max_wave_speed(UL), max_wave_speed(UR))
    return f - 0.5 * lam * (UR - UL)
