"""Euler-system MUSCL finite-volume solver — the FINE-GRID REFERENCE for the
hybrid-DGSEM alpha-policy training.

This is the 3-field (rho, rho*u, E) generalization of the scalar NN_MUSCL_Solver
in solver.py: component-wise minmod reconstruction + Rusanov (local
Lax-Friedrichs) numerical flux, SSP-RK2 in time, PERIODIC boundaries (np.roll).

It stays pure NumPy on purpose: the reference trajectory is a fixed target,
gradients never flow through it (Algorithm 1), so it does not need to be JAX.
The physics (gamma = 1.4, flux, wave speed) matches nn/jax_dgsem/physics.py
exactly so the MUSCL reference and the DGSEM trainee solve the SAME equations.

State convention matches the DGSEM package: U has shape (3, N) with
U[0]=rho, U[1]=rho*u, U[2]=E, on a uniform periodic grid of N finite-volume
cells with centers x_i = (i + 0.5) / N * L + xL.

Time stepping uses an IMPOSED dt (no CFL): dt is a solver-definition argument.
"""

import numpy as np

GAMMA = 1.4
GM1 = GAMMA - 1.0
EPS = 1e-12


def pressure(U):
    rho, rhou, E = U[0], U[1], U[2]
    return GM1 * (E - 0.5 * rhou * rhou / rho)


def euler_flux(U):
    """Euler flux F(U), shape (3, N) -> (3, N). Matches physics.physical_flux."""
    rho, rhou, E = U[0], U[1], U[2]
    u = rhou / rho
    p = GM1 * (E - 0.5 * rhou * u)
    return np.stack([rhou, rhou * u + p, u * (E + p)])


def max_wave_speed(U):
    """|u| + c per cell, shape (N,). Matches physics.max_wave_speed."""
    rho, rhou = U[0], U[1]
    u = rhou / rho
    p = pressure(U)
    return np.abs(u) + np.sqrt(np.maximum(GAMMA * p / rho, 0.0))


class MusclEulerGrid:
    """Uniform periodic finite-volume grid of N cells on [xL, xR].

    Cell centers x_i = xL + (i + 0.5) * dx (proper periodic FV grid, so x=xL
    and x=xR are the two faces of the wrap-around, never a duplicated node)."""

    def __init__(self, N: int, xL: float = 0.0, xR: float = 1.0):
        self.N = N
        self.xL = xL
        self.xR = xR
        self.L = xR - xL
        self.dx = self.L / N
        self.x = xL + (np.arange(N) + 0.5) * self.dx
        self.U = np.zeros((3, N))

    def set_state(self, U):
        self.U = np.asarray(U, dtype=np.float64).copy()


def _minmod(a, b):
    """Minmod of two slopes (elementwise)."""
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)),
                    0.0)


class MusclEulerSolver:
    """SSP-RK2 MUSCL-Rusanov solver on a periodic uniform grid, imposed dt."""

    def __init__(self, grid: MusclEulerGrid, dt: float, bc: str = "periodic"):
        """bc: 'periodic' (wrap, np.roll) or 'transmissive' (zero-gradient
        outflow, edge-clamped ghost) for the classic shock-tube cases."""
        self.grid = grid
        self.dx = grid.dx
        self.dt = dt
        self.bc = bc

    def _shift(self, A, direction):
        """A shifted along the cell axis: direction -1 -> A_{i+1}, +1 -> A_{i-1}.
        Periodic wraps; transmissive clamps to the edge cell (zero gradient)."""
        if self.bc == "periodic":
            return np.roll(A, direction, axis=1)
        if direction == -1:                             # A_{i+1}
            return np.concatenate([A[:, 1:], A[:, -1:]], axis=1)
        return np.concatenate([A[:, :1], A[:, :-1]], axis=1)   # A_{i-1}

    def compute_rhs(self, U):
        """-dF/dx via component-wise minmod reconstruction + Rusanov flux.

        Boundaries via _shift: periodic (wrap) or transmissive (edge-clamp)."""
        du_fwd = self._shift(U, -1) - U                # U_{i+1} - U_i
        du_bwd = U - self._shift(U, 1)                 # U_i - U_{i-1}
        slope = _minmod(du_bwd, du_fwd)                # limited cell slope

        # reconstructed states at interface i+1/2:
        #   left  from cell i   : U_i   + 0.5 slope_i
        #   right from cell i+1 : U_{i+1} - 0.5 slope_{i+1}
        U_L = U + 0.5 * slope
        U_R = self._shift(U, -1) - 0.5 * self._shift(slope, -1)

        F_L = euler_flux(U_L)
        F_R = euler_flux(U_R)
        a = np.maximum(max_wave_speed(U_L), max_wave_speed(U_R))   # (N,)

        # Rusanov flux at every interface i+1/2, shape (3, N)
        flux = 0.5 * (F_L + F_R) - 0.5 * a[None, :] * (U_R - U_L)

        # net difference F_{i+1/2} - F_{i-1/2}
        flux_diff = flux - self._shift(flux, 1)
        return -flux_diff / self.dx

    def step(self):
        """One SSP-RK2 (Heun) step with the imposed dt."""
        U_n = self.grid.U
        U_1 = U_n + self.dt * self.compute_rhs(U_n)
        self.grid.U = 0.5 * U_n + 0.5 * (U_1 + self.dt * self.compute_rhs(U_1))
        return self.grid.U

    def run(self, n_steps: int):
        """Advance n_steps and return the final state (3, N)."""
        for _ in range(n_steps):
            self.step()
        return self.grid.U

    def trajectory(self, n_outer: int, inner_steps: int):
        """Advance n_outer * inner_steps steps, snapshotting the state every
        inner_steps steps. Returns (n_outer + 1, 3, N) including the IC.

        Used to sample the reference at the DGSEM coarse-step cadence: one
        outer snapshot per DGSEM step, inner_steps MUSCL substeps in between.
        """
        out = np.empty((n_outer + 1, 3, self.grid.N))
        out[0] = self.grid.U
        for k in range(n_outer):
            for _ in range(inner_steps):
                self.step()
            out[k + 1] = self.grid.U
        return out
