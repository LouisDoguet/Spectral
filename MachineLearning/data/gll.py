"""GLL nodes, weights, and derivative matrix — mirrors lib/base/gll.h."""
import numpy as np


def _legendre_and_deriv(xi, P):
    """Evaluate P_j(xi) and dP_j/dxi for j = 0..P via recurrence."""
    n = len(xi)
    L  = np.zeros((n, P + 1))
    dL = np.zeros((n, P + 1))
    L[:, 0] = 1.0
    if P >= 1:
        L[:, 1] = xi
        dL[:, 1] = 1.0
    for j in range(2, P + 1):
        L[:, j]  = ((2*j - 1) * xi * L[:, j-1] - (j-1) * L[:, j-2]) / j
        dL[:, j] = ((2*j - 1) * (L[:, j-1] + xi * dL[:, j-1]) - (j-1) * dL[:, j-2]) / j
    return L, dL


def nodes_weights(P):
    """
    Gauss-Legendre-Lobatto nodes (ascending in [-1, 1]) and weights.
    Returns (xi, w) both of shape (P+1,).
    """
    n = P + 1
    xi = -np.cos(np.pi * np.arange(n) / P)  # Chebyshev-Gauss-Lobatto initial guess
    xi[0], xi[-1] = -1.0, 1.0

    for _ in range(100):
        L, dL = _legendre_and_deriv(xi, P)
        # Interior GLL nodes are zeros of q(xi) = xi*P_P - P_{P-1}
        q  =  xi * L[:, P] - L[:, P - 1]
        dq = L[:, P] + xi * dL[:, P] - dL[:, P - 1]
        delta = np.zeros(n)
        delta[1:-1] = q[1:-1] / dq[1:-1]
        xi -= delta
        if np.max(np.abs(delta)) < 1e-15:
            break

    L, _ = _legendre_and_deriv(xi, P)
    w = 2.0 / (P * (P + 1) * L[:, P] ** 2)
    return xi, w


def deriv_matrix(xi, P):
    """
    Spectral derivative matrix D on GLL nodes.
    D[i, j] = dL_j/dxi evaluated at xi_i, where L_j is the j-th Lagrange basis.
    (D @ f)[i] gives the derivative of f at node i in reference space [-1, 1].)
    """
    n = P + 1
    L, _ = _legendre_and_deriv(xi, P)
    LP = L[:, P]

    xi_i = xi[:, np.newaxis]
    xi_j = xi[np.newaxis, :]
    LP_i = LP[:, np.newaxis]
    LP_j = LP[np.newaxis, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(xi_i != xi_j, LP_i / (LP_j * (xi_i - xi_j)), 0.0)

    D[0, 0] = -P * (P + 1) / 4.0
    D[P, P] =  P * (P + 1) / 4.0
    return D
