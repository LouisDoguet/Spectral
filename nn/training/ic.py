"""Shared periodic random-Fourier Euler initial conditions.

One analytic function of x, sampled natively on BOTH grids (the fine MUSCL
reference grid and the coarse DGSEM trainee grid), so the two solvers are
handed the exact same problem.

The rho / p fields are bounded, strictly positive, moderate-amplitude periodic
Fourier series:

    rho(x) = rho0 + amp * s_rho(x) / max|s_rho|      (in [rho0-amp, rho0+amp])
    p(x)   = p0   + amp * s_p(x)   / max|s_p|

Normalizing each series by its max-abs (computed once on a fixed dense grid, so
it is a property of the function and identical on both grids) keeps rho, p
strictly positive and keeps the sound speed bounded -- essential because the
training uses an IMPOSED dt (no CFL).

The VELOCITY is controlled to force a discontinuity to develop within the
rollout. Nonlinear steepening turns a velocity COMPRESSION (a region with
du/dx < 0) into a shock at the Burgers-like time t_shock ~ 1 / |min(du/dx)|.
So instead of just scaling the velocity by an amplitude, we rescale it so its
strongest compression equals a target value C:

    u(x) = k * (s_u(x) - mean s_u),   k chosen so  min(du/dx) = -C

Empirically (MUSCL Euler, calibrated) the shock forms fully in ~ SHOCK_CALIB / C
seconds, i.e. C = SHOCK_CALIB / t_shock; compression_for_shock_fraction() maps a
"form the shock at fraction f of the rollout" request to C. Larger C => faster
shock (but larger velocity => higher wave speed, watch the imposed-dt CFL).
"""

import numpy as np

GAMMA = 1.4
_DENSE = 4096          # dense grid for the grid-independent normalization
SHOCK_CALIB = 1.2      # C * t_shock, fitted to the MUSCL Euler solver


def compression_for_shock_fraction(fraction: float, rollout_steps: int,
                                   dt: float) -> float:
    """Velocity compression C that forms a shock at ~`fraction` of the rollout.

    fraction in (0, 1): 0.6 means the discontinuity develops around 60% of the
    way through, leaving room to be fully formed by the final step."""
    t_shock = max(fraction, 1e-6) * rollout_steps * dt
    return SHOCK_CALIB / t_shock


def draw_fourier_coeffs(rng: np.random.Generator, n_modes: int = 12,
                        amp: float = 0.4, xL: float = 0.0, xR: float = 1.0,
                        rho0: float = 1.0, p0: float = 1.0,
                        amp_u: float = None, target_compression: float = None,
                        vel_modes: int = 3):
    """Random coefficients for one IC + baked-in per-field normalizations.

    amp   : half-amplitude of rho and p about rho0 / p0 (must be < rho0, p0).
    amp_u : velocity amplitude, used only when target_compression is None.
    target_compression C : if set, the velocity is rescaled so its strongest
        compression is min(du/dx) = -C, forcing a shock to develop in
        ~ SHOCK_CALIB / C seconds (see compression_for_shock_fraction).
    vel_modes : number of velocity Fourier modes. Deliberately small (few, low
        wavenumbers) so the compression is broad and coherent and a shock forms
        reliably every draw; rho/p keep the richer n_modes for diversity.
    """
    if amp_u is None:
        amp_u = amp
    coeffs = {"n_modes": n_modes, "xL": xL, "xR": xR,
              "rho0": rho0, "p0": p0, "amp": amp, "amp_u": amp_u}
    x_dense = xL + (np.arange(_DENSE) + 0.5) * (xR - xL) / _DENSE
    for field, nm in (("rho", n_modes), ("p", n_modes), ("u", vel_modes)):
        a = rng.uniform(0.0, 1.0, nm)
        b = rng.uniform(0.0, 1.0, nm)
        coeffs[field] = (a, b)
        raw = _series((a, b), x_dense, coeffs)
        coeffs[field + "_norm"] = max(float(np.max(np.abs(raw))), 1e-12)

    # Velocity scaling. Default: amplitude-normalized like rho/p. With a target
    # compression: centre the field and rescale so its steepest slope is a
    # compression (du/dx < 0) of magnitude C.
    u_raw = _series(coeffs["u"], x_dense, coeffs)
    u_mean = float(u_raw.mean())
    coeffs["u_center"] = u_mean
    if target_compression is None:
        coeffs["u_scale"] = amp_u / coeffs["u_norm"]
    else:
        dudx = np.gradient(u_raw - u_mean, x_dense)
        gmin, gmax = float(dudx.min()), float(dudx.max())
        # orient so the steepest slope becomes the -C compression
        if abs(gmin) >= abs(gmax):
            coeffs["u_scale"] = target_compression / abs(gmin)
        else:
            coeffs["u_scale"] = -target_compression / abs(gmax)
    return coeffs


def _series(ab, x, coeffs):
    """sum_n a_n/(n+1) cos(2 pi n (x-xL)/L) + b_n/(n+1) sin(...).

    The mode count is inferred from len(a), so rho/p (n_modes) and the velocity
    (vel_modes, deliberately fewer -> a broad low-wavenumber compression that
    steepens into a shock reliably) can share the same evaluator."""
    a, b = ab
    L = coeffs["xR"] - coeffs["xL"]
    xn = (np.asarray(x) - coeffs["xL"]) / L
    n = np.arange(len(a))
    phases = 2.0 * np.pi * n[:, None] * xn[None, :]     # (len(a), Nx)
    scale = 1.0 / (n + 1.0)
    return (a * scale) @ np.cos(phases) + (b * scale) @ np.sin(phases)


def eval_primitives(coeffs, x):
    """Evaluate (rho, u, p) at arbitrary points x (any shape via ravel)."""
    xflat = np.asarray(x, dtype=np.float64).ravel()
    amp = coeffs["amp"]
    rho = coeffs["rho0"] + amp * _series(coeffs["rho"], xflat, coeffs) \
        / coeffs["rho_norm"]
    p = coeffs["p0"] + amp * _series(coeffs["p"], xflat, coeffs) \
        / coeffs["p_norm"]
    # velocity: k * (series - mean), with k / mean baked in at draw time so the
    # compression is a grid-independent property of the function
    u = coeffs["u_scale"] * (_series(coeffs["u"], xflat, coeffs)
                             - coeffs["u_center"])
    shape = np.asarray(x).shape
    return rho.reshape(shape), u.reshape(shape), p.reshape(shape)


def primitives_to_conservative(rho, u, p):
    """(rho, u, p) -> (rho, rho*u, E) stacked on a leading axis of size 3."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E])


def sample_on_muscl(coeffs, grid):
    """Conservative state (3, N) on a MusclEulerGrid's cell centers."""
    rho, u, p = eval_primitives(coeffs, grid.x)
    return primitives_to_conservative(rho, u, p)


def sample_on_dgsem(coeffs, mesh, xL: float = 0.0):
    """Conservative state (3, n_elem, Nn) on the DGSEM GLL nodes."""
    x = np.asarray(mesh.node_positions(xL))       # (n_elem, Nn)
    rho, u, p = eval_primitives(coeffs, x)
    return primitives_to_conservative(rho, u, p)
