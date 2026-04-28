"""Per-element optimal-eps search against the exact Sod solution.

For each element at each snapshot, we find the scalar eps_e in [0, eps_max]
that minimises the L2 error between the one-step SEM solution and the exact
solution.  Elements are independent so the search is embarrassingly parallel.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from data.sod_exact import sod_exact, GAMMA
from labels.sem1d import step


def compute_labels(
    snapshot: dict,
    dt: float,
    geom: dict,
    bc_L: tuple,
    bc_R: tuple,
    eps_max: float = 0.5,
    sod_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    snapshot   : dict from data.loader.load_snapshot
    dt         : timestep used by the C++ solver
    geom       : dict from labels.sem1d.make_geometry
    bc_L, bc_R : (rho, rhou, e) Dirichlet boundary state tuples
    eps_max    : upper bound for the element-wise search
    sod_kwargs : extra keyword arguments forwarded to sod_exact
                 (e.g. x0, rho_L, p_L, ...)

    Returns
    -------
    eps_labels : (n_total,) array  — same eps value for all nodes in an element
    """
    if sod_kwargs is None:
        sod_kwargs = {}

    rho, rhou, enrg = snapshot["rho"], snapshot["rhou"], snapshot["enrg"]
    t       = snapshot["t"]
    n_elem  = geom["n_elem"]
    n       = geom["n"]       # P + 1
    n_total = geom["n_total"]
    x       = geom["x"]

    # Exact conserved variables at t + dt
    rho_ex, u_ex, p_ex = sod_exact(x, t + dt, **sod_kwargs)
    enrg_ex = p_ex / (GAMMA - 1) + 0.5 * rho_ex * u_ex ** 2
    U_exact = np.stack([rho_ex, rho_ex * u_ex, enrg_ex])  # (3, n_total)

    eps_labels = np.zeros(n_total)

    for elem_idx in range(n_elem):
        sl = slice(elem_idx * n, (elem_idx + 1) * n)

        def loss(eps_e, sl=sl):
            eps_array = np.zeros(n_total)
            eps_array[sl] = eps_e
            rho_n, rhou_n, enrg_n = step(rho, rhou, enrg, eps_array, dt,
                                          geom, bc_L, bc_R)
            err = ((rho_n[sl]   - U_exact[0, sl]) ** 2 +
                   (rhou_n[sl]  - U_exact[1, sl]) ** 2 +
                   (enrg_n[sl]  - U_exact[2, sl]) ** 2)
            return float(np.mean(err))

        result = minimize_scalar(loss, bounds=(0.0, eps_max), method="bounded")
        eps_labels[sl] = result.x

    return eps_labels
