"""
Burgers equation: du/dt + d(u^2/2)/dx = 0

Encapsulates the PDE definition (flux, exact solution) together with the
RBF-based DG residual R = M^{-1} @ (D @ F(u)) used as the optimization
objective in optimize.py.
"""

import numpy as np


class Burgers:
    """
    Parameters
    ----------
    P : int
        Polynomial order (number of nodes = P+1).
    shock_intensity : float
        Steepness of the tanh profile used by `exact_solution`.
    """

    def __init__(self, P=10, shock_intensity=10.0):
        self.P = P
        self.shock_intensity = shock_intensity

    def exact_solution(self, x, shock_pos=0.0, u_left=-1.0, u_right=1.0):
        """Tanh shock profile, for validation/plotting only."""
        x = np.asarray(x, dtype=float)
        return (u_left + u_right) / 2 + (u_right - u_left) / 2 * np.tanh(
            self.shock_intensity * (x - shock_pos)
        )

    def flux(self, u):
        """Burgers flux F(u) = u^2 / 2."""
        return 0.5 * np.asarray(u) ** 2

    def _collocation_matrix(self, X, eps_array):
        """A[i,j] = exp(-(eps_j * |x_i - x_j|)^2)"""
        r = np.abs(X[:, None] - X[None, :])
        return np.exp(-(eps_array[None, :] * r) ** 2)

    def collocation_condition_number(self, X, eps_array):
        """cond(A) for the RBF collocation matrix, used as an ill-conditioning penalty."""
        X = np.asarray(X, dtype=float)
        eps_array = np.asarray(eps_array, dtype=float)
        A = self._collocation_matrix(X, eps_array)
        return np.linalg.cond(A)

    def compute_rbf_derivative_matrix(self, X, eps_array, verbose=False):
        """
        Exact RBF derivative matrix D such that u'(x_i) = sum_j D[i,j] * u(x_j),
        built from the closed-form derivative of the Gaussian kernel.
        """
        X = np.asarray(X, dtype=float)
        eps_array = np.asarray(eps_array, dtype=float)

        A = self._collocation_matrix(X, eps_array)
        cond = np.linalg.cond(A)
        if cond > 1e10:
            if verbose:
                print(f"WARNING: Collocation matrix ill-conditioned (cond={cond:.2e})")
            if cond > 1e15:
                raise np.linalg.LinAlgError("Collocation matrix is singular")
        invA = np.linalg.inv(A)

        # Signed pairwise differences and the derivative of phi w.r.t. r
        diff = X[:, None] - X[None, :]
        abs_diff = np.abs(diff)
        eps_j = eps_array[None, :]
        phi_prime = -2 * eps_j**2 * abs_diff * np.exp(-(eps_j * abs_diff) ** 2)
        D_kernel = np.sign(diff) * phi_prime

        return D_kernel @ invA

    def compute_rbf_mass_matrix(self, X, eps_array, quad_order=None, verbose=False):
        """
        RBF mass matrix M[i,j] = integral_{-1}^{1} l_i(x) l_j(x) dx, via
        Gauss-Legendre quadrature on the RBF Lagrange basis l = invA @ phi.
        """
        X = np.asarray(X, dtype=float)
        eps_array = np.asarray(eps_array, dtype=float)

        if quad_order is None:
            quad_order = max(3 * len(X), 64)

        xi_quad, w_quad = np.polynomial.legendre.leggauss(quad_order)

        A = self._collocation_matrix(X, eps_array)
        invA = np.linalg.inv(A)

        dist = np.abs(xi_quad[None, :] - X[:, None])
        phi = np.exp(-(eps_array[:, None] * dist) ** 2)
        L_rbf = invA @ phi

        M = L_rbf @ np.diag(w_quad) @ L_rbf.T

        if verbose:
            symm_error = np.max(np.abs(M - M.T))
            if symm_error > 1e-10:
                print(f"WARNING: Mass matrix not symmetric (error={symm_error:.2e})")

            eigs = np.linalg.eigvalsh(M)
            if np.min(eigs) < -1e-10:
                print(f"WARNING: Mass matrix near singular (min eigenvalue={np.min(eigs):.2e})")

        return M

    def compute_residual(self, u_values, X, eps_array, verbose=False):
        """DG weak-form residual R = M^{-1} @ (D @ F(u))."""
        X = np.asarray(X, dtype=float)
        eps_array = np.asarray(eps_array, dtype=float)

        F = self.flux(u_values)
        D = self.compute_rbf_derivative_matrix(X, eps_array, verbose=verbose)
        divF = D @ F
        M = self.compute_rbf_mass_matrix(X, eps_array, verbose=verbose)

        return np.linalg.solve(M, divF)

    def residual_norm_squared(self, u_values, X, eps_array):
        """Optimization objective ||R||^2; returns inf on singular configurations."""
        try:
            R = self.compute_residual(u_values, X, eps_array)
            return np.sum(R**2)
        except np.linalg.LinAlgError:
            return np.inf
