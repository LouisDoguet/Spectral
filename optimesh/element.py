import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x: float, val: float = 0.0):
        self.x = x
        self.value = val

class Element:
    def __init__(self, P: int, uL: float, uR: float, shock_pos: float, shock_intensity: float):
        if abs(shock_pos) > 1: 
            raise ValueError("Shock position not in the element")
        
        self.P = P
        self.uL = uL
        self.uR = uR
        self.shock_pos = shock_pos
        self.shock_intensity = shock_intensity

        self.eps_cluster = None
        
        self.discontinuity = lambda x: (uL + uR) / 2 + abs(uL - uR) / 2 * -np.tanh(shock_intensity * (x - shock_pos))
        
        self.X = np.linspace(-1, 1, P + 1)
        self.val = self.discontinuity(self.X)
        
        self.nodes = [Node(x, v) for x, v in zip(self.X, self.val)]

    def plot(self, disp=True):
        fig, axes = plt.subplots(figsize=(8, 4))
        
        # Plot a dense continuous curve in the background for reference
        x_dense = np.linspace(-1, 1, 500)
        axes.plot(x_dense, self.discontinuity(x_dense), 'k--', alpha=0.5, label='Exact Function')
        
        # Plot our actual nodes
        axes.scatter(self.X, np.zeros_like(self.X), color='red', marker='x', label='Node Positions (y=0)')
        axes.plot(self.X, self.val, 'bo-', label='Sampled Grid')
        
        axes.set_xlim(-1.1, 1.1)
        axes.legend()
        axes.grid(True)
        if disp: plt.show()
        return fig, axes

    def display(self, solution):
        fig, axes = self.plot(False)
        axes.plot(solution.xi, solution.values, label="RBF solution")
        axes.legend()
        plt.show()

    def local_spacing(self) -> np.ndarray:
        """
        Returns the local mesh spacing h_i at each node, defined as the
        average distance to its neighbours (one-sided at the boundaries).
        Used to adapt the RBF shape parameter to local clustering.
        """
        h = np.empty(self.P + 1)
        h[0] = self.X[1] - self.X[0]
        h[-1] = self.X[-1] - self.X[-2]
        if self.P > 1:
            h[1:-1] = (self.X[2:] - self.X[:-2]) / 2
        return h

    def cluster(self, x0: float, epsilon: float):
        """
        Clusters nodes tightly around x0 with intensity epsilon.
        Uses the inverse hyperbolic tangent (arctanh) mapping to concentrate
        grid density at x0 while preserving the exact boundaries [-1, 1].
        """
        self.eps_cluster = epsilon
        if epsilon < 1e-5:
            self.X = np.linspace(-1, 1, self.P + 1)
        else:
            # 1. Find the boundaries in the transformed tanh space
            q_left = np.tanh(epsilon * (-1 - x0))
            q_right = np.tanh(epsilon * (1 - x0))
            
            # 2. Take uniform steps in the transformed space
            q_uniform = np.linspace(q_left, q_right, self.P + 1)
            
            # 3. Clip infinitesimally inside the domain to prevent float64 warnings
            # This protects against np.arctanh(1.0) yielding infinity if epsilon is large
            q_uniform = np.clip(q_uniform, -1 + 1e-15, 1 - 1e-15)
            
            # 4. Map back to the physical space using arctanh
            self.X = x0 + (1.0 / epsilon) * np.arctanh(q_uniform)
            
            # 5. Strictly enforce absolute boundaries to fix floating-point drift
            self.X[0] = -1.0
            self.X[-1] = 1.0

        # Update values and the internal node objects
        self.val = self.discontinuity(self.X)
        for i in range(self.P + 1):
            self.nodes[i].x = self.X[i]
            self.nodes[i].value = self.val[i]
