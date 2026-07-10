"""Reference-trajectory generation and initial conditions (paper Sect. 3.4).

The reference scheme S_ref is the SAME entropy-stable hybrid DGSEM but on a
refine-times finer mesh, driven by the built-in Persson-Peraire indicator and
advanced with dt/refine (accurate + robust, mirroring how reference_data.job
produces training data with the C++ solver). Gradients never flow through it:
its output is a fixed target, exactly like the paper's MUSCL reference.

Also provides a loader for the C++ binary snapshots (solver.cpp
export_snapshot), so C++-generated references can be swapped in later.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.basis import gll_nodes, lagrange_interpolation_matrix
from jax_dgsem.solver import rk4_step
from jax_dgsem.indicator import persson_peraire_alpha

GAMMA = 1.4


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def primitives_to_conservative(rho, u, p):
    return jnp.stack([rho, rho * u, p / (GAMMA - 1.0) + 0.5 * rho * u * u])


def _fourier_series(key, x01, n_modes):
    """sum_n a_n/(n+1) cos(2 pi n x) + b_n/(n+1) sin(2 pi n x), a,b ~ U[-1,1]
    (paper Eq. 9 / muscl/case.py)."""
    ka, kb = jax.random.split(key)
    a = jax.random.uniform(ka, (n_modes,), minval=-1.0, maxval=1.0)
    b = jax.random.uniform(kb, (n_modes,), minval=-1.0, maxval=1.0)
    n = jnp.arange(n_modes)
    phases = 2.0 * jnp.pi * n[:, None] * x01[None, :]
    scale = 1.0 / (n + 1.0)
    return (a * scale) @ jnp.cos(phases) + (b * scale) @ jnp.sin(phases)


def random_fourier_ic(key, mesh: Mesh1D, xL: float, n_modes: int = 20,
                      eps: float = 0.1):
    """Random smooth IC on the primitive variables (rho, u, p), with the
    positivity shift g - min(g) + eps applied to rho and p (Sect. 3.4)."""
    x = mesh.node_positions(xL)
    x01 = ((x - xL) / (mesh.dx * mesh.n_elem)).ravel()
    k1, k2, k3 = jax.random.split(key, 3)
    shape = x.shape

    def positive(k):
        g = _fourier_series(k, x01, n_modes)
        return (g - jnp.min(g) + eps).reshape(shape)

    rho = positive(k1)
    p = positive(k3)
    u = _fourier_series(k2, x01, n_modes).reshape(shape)
    return primitives_to_conservative(rho, u, p)


def riemann_ic(mesh: Mesh1D, xL: float, left, right, x0: float,
               delta: float = None):
    """Tanh-smoothed two-state IC (SOD.cpp generateMesh). left/right are
    primitive (rho, u, p) tuples; delta defaults to 2*dx."""
    if delta is None:
        delta = 2.0 * mesh.dx
    x = mesh.node_positions(xL)
    s = 0.5 * (1.0 - jnp.tanh((x - x0) / delta))
    rho = right[0] + (left[0] - right[0]) * s
    u = right[1] + (left[1] - right[1]) * s
    p = right[2] + (left[2] - right[2]) * s
    return primitives_to_conservative(rho, u, p)


SOD = dict(left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), x0=0.5,
           domain=(0.0, 1.0))
LAX = dict(left=(0.445, 0.698, 3.528), right=(0.5, 0.0, 0.571), x0=0.5,
           domain=(0.0, 1.0))


def shu_osher_ic(mesh: Mesh1D, xL: float = -5.0, delta: float = None):
    """Shu-Osher shock/entropy-wave interaction on [-5,5] (shu_osher.cpp):
    post-shock state (3.857143, 2.629369, 10.33333) for x < -4, entropy wave
    (1 + 0.2 sin 5x, 0, 1) on the right, tanh-blended over delta (def 2*dx)."""
    if delta is None:
        delta = 2.0 * mesh.dx
    x = mesh.node_positions(xL)
    rhoL, uL, pL = 3.857143, 2.629369, 10.33333
    rhoR = 1.0 + 0.2 * jnp.sin(5.0 * x)
    s = 0.5 * (1.0 - jnp.tanh((x - (-4.0)) / delta))
    rho = rhoR + (rhoL - rhoR) * s
    u = uL * s
    p = 1.0 + (pL - 1.0) * s
    return primitives_to_conservative(rho, u, p)


def random_riemann_ic(key, mesh: Mesh1D, xL: float):
    """Randomized shock tube: a two-state Riemann IC with Sod/Lax-like
    magnitudes (Sod itself lies inside this family). Tanh-smoothed over 2*dx
    like the C++ cases. Guarantees a steep feature inside every trajectory,
    so an under-stabilized policy is punished within the training window."""
    ks = jax.random.split(key, 7)
    u = lambda k, lo, hi: jax.random.uniform(k, minval=lo, maxval=hi)
    left = (u(ks[0], 0.6, 1.4), u(ks[1], -0.3, 0.8), u(ks[2], 0.6, 1.4))
    right = (u(ks[3], 0.1, 0.5), u(ks[4], -0.3, 0.3), u(ks[5], 0.05, 0.5))
    L = mesh.dx * mesh.n_elem
    x0 = xL + L * u(ks[6], 0.35, 0.65)
    return riemann_ic(mesh, xL, left, right, x0)


def wall_states_from_ic(U0):
    """Fixed Dirichlet ghost states = the IC endpoint states (how the C++
    cases set their bc::Wall)."""
    return U0[:, 0, 0], U0[:, -1, -1]


# ---------------------------------------------------------------------------
# Mesh transfer operators
# ---------------------------------------------------------------------------

def refine_ic(U_coarse, P: int, refine: int):
    """Interpolate a coarse DG field onto the refine-x finer mesh (exact:
    same piecewise polynomial, resampled)."""
    q = gll_nodes(P)
    blocks = []
    for s in range(refine):
        xi_f = (q + 1.0) / refine + s * 2.0 / refine - 1.0  # fine nodes in coarse elem
        blocks.append(lagrange_interpolation_matrix(q, xi_f))
    T = jnp.asarray(np.stack(blocks))  # (refine, Nn_f, Nn_c)
    vals = jnp.einsum("sfj,cej->cesf", T, U_coarse)
    return vals.reshape(U_coarse.shape[0], -1, len(q))


def restriction_operator(P: int, refine: int):
    """Matrix B (Nn_c, refine, Nn_f): coarse-node values from the fine
    piecewise polynomial (used to start sub-trajectories from the reference,
    Algorithm 1 line 7)."""
    q = gll_nodes(P)
    B = np.zeros((P + 1, refine, P + 1))
    for i, xi in enumerate(q):
        t = (xi + 1.0) / 2.0 * refine
        s = min(int(t), refine - 1)
        xi_f = 2.0 * (t - s) - 1.0
        B[i, s, :] = lagrange_interpolation_matrix(q, np.array([xi_f]))[0]
    return jnp.asarray(B)


def restrict(U_fine, B, n_elem_coarse: int):
    """(3, n_c*refine, Nn) -> (3, n_c, Nn) using B from restriction_operator."""
    refine = U_fine.shape[1] // n_elem_coarse
    Uf = U_fine.reshape(U_fine.shape[0], n_elem_coarse, refine, -1)
    return jnp.einsum("isj,cesj->cei", B, Uf)


# ---------------------------------------------------------------------------
# Reference trajectories
# ---------------------------------------------------------------------------

def generate_reference_trajectory(U0_fine, mesh_fine: Mesh1D, n_steps: int,
                                  dt_coarse: float, refine: int,
                                  alpha_max: float = 1.0):
    """Run the PP-driven hybrid solver on the fine mesh with dt/refine and
    return the states at every COARSE step time: (n_steps+1, 3, n_f, Nn).

    jitted + lax.scan; call inside jax.lax.stop_gradient territory (no grads
    are taken through this anyway since its output is only used as data).
    """
    dt_f = dt_coarse / refine

    def alpha_fn(U):
        return persson_peraire_alpha(U, mesh_fine, alpha_max=alpha_max)

    def fine_step(U, _):
        U1 = rk4_step(U, alpha_fn(U), dt_f, mesh_fine)
        return U1, None

    def coarse_step(U, _):
        U1, _ = jax.lax.scan(fine_step, U, None, length=refine)
        return U1, U1

    _, traj = jax.lax.scan(coarse_step, U0_fine, None, length=n_steps)
    return jnp.concatenate([U0_fine[None], traj], axis=0)


# ---------------------------------------------------------------------------
# C++ binary snapshots (solver.cpp export_snapshot)
# ---------------------------------------------------------------------------

def load_cpp_snapshot(path):
    """Read one snap_XXXXXX.bin: returns (U, time) with U (3, n_elem, P+1)."""
    with open(path, "rb") as f:
        n_elem, P = np.fromfile(f, dtype=np.int32, count=2)
        time = np.fromfile(f, dtype=np.float64, count=1)[0]
        total = n_elem * (P + 1)
        fields = [np.fromfile(f, dtype=np.float64, count=total) for _ in range(3)]
    U = np.stack(fields).reshape(3, n_elem, P + 1)
    return jnp.asarray(U), float(time)
