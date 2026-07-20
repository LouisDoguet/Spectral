"""Gauss-Legendre-Lobatto basis, replicating lib/base/gll.cpp bit-for-bit.

Everything here is setup-time NumPy (float64); the results are stored as jnp
arrays on the GLLBasis object and treated as constants by autograd.
"""

import numpy as np
import jax.numpy as jnp


def legendre(P: int, xi):
    """Legendre polynomial L_P(xi) by the Bonnet recursion (gll.cpp Bonnet)."""
    xi = np.asarray(xi, dtype=np.float64)
    if P == 0:
        return np.ones_like(xi)
    if P == 1:
        return xi.copy()
    L_nm2 = np.ones_like(xi)
    L_nm1 = xi.copy()
    for n in range(2, P + 1):
        L_n = ((2.0 * n - 1.0) * xi * L_nm1 - (n - 1.0) * L_nm2) / n
        L_nm2, L_nm1 = L_nm1, L_n
    return L_n


def _Lp(P, xi):
    """d/dxi L_P(xi) (gll.cpp Lpp)."""
    return (P / (1.0 - xi * xi)) * (legendre(P - 1, xi) - xi * legendre(P, xi))


def _Lpp(P, xi):
    """d2/dxi2 L_P(xi) (gll.cpp Lppp)."""
    return -(P * (P + 1) / (1.0 - xi * xi)) * legendre(P, xi)


def gll_nodes(P: int) -> np.ndarray:
    """GLL nodes on [-1,1]: endpoints plus roots of L'_P (gll.cpp setQuads)."""
    q = np.zeros(P + 1)
    q[0], q[P] = -1.0, 1.0
    for k in range(1, (P + 1) // 2 + 1):
        xi = -np.cos(np.pi * k / P)
        for _ in range(50):
            step = _Lp(P, xi) / _Lpp(P, xi)
            xi_new = xi - step
            if abs(xi_new - xi) < 1e-15:
                xi = xi_new
                break
            xi = xi_new
        q[k] = xi
        q[P - k] = -xi
    return q


def gll_weights(P: int, quads: np.ndarray) -> np.ndarray:
    """w_i = 2 / (P(P+1) L_P(x_i)^2) (gll.cpp setWeights)."""
    Lpi = legendre(P, quads)
    return 2.0 / (P * (P + 1) * Lpi * Lpi)


def derivative_matrix(P: int, quads: np.ndarray) -> np.ndarray:
    """Nodal derivative matrix with negative-sum diagonal (gll.cpp)."""
    N = P + 1
    D = np.zeros((N, N))
    L = legendre(P, quads)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (L[i] / L[j]) / (quads[i] - quads[j])
        D[i, i] = -np.sum(D[i, :])
    return D


def modal_projection_matrix(P: int, quads: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
    """Nodal -> Legendre-modal matrix (math.cpp computeLegendreCoeffs).

    c_k = (2k+1)/2 * sum_j u_j L_k(x_j) w_j   =>   c = Vm @ u
    """
    N = P + 1
    Vm = np.zeros((N, N))
    for k in range(N):
        Vm[k, :] = (2.0 * k + 1.0) / 2.0 * legendre(k, quads) * weights
    return Vm


def lagrange_interpolation_matrix(quads: np.ndarray,
                                  xi_out: np.ndarray) -> np.ndarray:
    """Cardinal Lagrange interpolation from GLL nodes to points xi_out.

    Returns T with u(xi_out) = T @ u_nodes (base.cpp Lagrange::interpolate).
    """
    N = len(quads)
    T = np.ones((len(xi_out), N))
    for i in range(N):
        for j in range(N):
            if i != j:
                T[:, i] *= (xi_out - quads[j]) / (quads[i] - quads[j])
    return T


class GLLBasis:
    """Immutable container of the reference-element operators (jnp arrays)."""

    def __init__(self, P: int):
        self.P = P
        self.N = P + 1
        q = gll_nodes(P)
        w = gll_weights(P, q)
        self.quads = jnp.asarray(q)
        self.weights = jnp.asarray(w)
        self.D = jnp.asarray(derivative_matrix(P, q))
        self.modal = jnp.asarray(modal_projection_matrix(P, q, w))

    def interpolation_to(self, xi_out: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            lagrange_interpolation_matrix(np.asarray(self.quads), xi_out))
    


P = 3
b = GLLBasis(P)
print(b.D)
print(b.weights)

