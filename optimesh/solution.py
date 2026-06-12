import numpy as np
import matplotlib.pyplot as plt
import element as el

class Solution:
    def __init__(self, element:el.Element, eps:float, rbf = lambda r, eps: np.exp(-eps * r**2)):
        self.rbf = rbf
        self.xi = np.linspace(-1, 1, 1000)
        self.X = element.X

        # Adapt each node's shape parameter to the local mesh spacing so that
        # clustered nodes (small h) get narrower RBFs and sparse nodes (large h)
        # get wider ones, keeping eps_j * h_j^2 close to eps * h0^2 (the value
        # eps would have on a uniform mesh with the same number of nodes).
        h = element.local_spacing()
        h0 = 2.0 / element.P
        self.eps = eps * (h0 / h) ** 2

        self.R = np.abs(self.X[:, None] - self.X[None, :])
        self.A = rbf(self.R, self.eps[None, :])
        self.invA = np.linalg.inv(self.A)

        dist_xi = np.abs(self.xi[None, :] - self.X[:, None])
        self.phi = rbf(dist_xi, self.eps[:, None])

        self.c = self.invA @ element.val
        self.values = self.c @ self.phi

    def plot_rbfs(self, ax=None, disp=True):
        """
        Plot the individual RBF associated with each node together with the
        resulting interpolant, illustrating how each node's shape parameter
        (radius) adapts to the local clustering.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        for j in range(len(self.X)):
            ax.plot(self.xi, self.phi[j, :], alpha=0.6,
                    label=f"node {j} ($\\epsilon$={self.eps[j]:.2f})")

        ax.scatter(self.X, np.ones_like(self.X), color='k', marker='x', zorder=5)
        ax.plot(self.xi, self.values, 'k-', linewidth=2, label="RBF solution")

        ax.set_xlim(-1.1, 1.1)
        ax.grid(True)
        ax.legend(fontsize='x-small', ncol=2)
        ax.set_title("RBF basis functions at each node")

        if disp:
            plt.show()
        return fig, ax