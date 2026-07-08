import numpy as np
import matplotlib.pyplot as plt

EPS = 10e-3

class NN_Mesh:
    def __init__(self, N: int = 2048, n_batch: int = 1):
        self.n_batch = n_batch
        self.N = N
        
        self.X = np.linspace(0, 1, N) 
        
        # Batched Values
        self.VAL = np.zeros((n_batch, N))

    def setInitialCondition(self, MaxFourrierExp: int = 20, rand_seed: int = 0):
        np.random.seed(rand_seed)
        
        # Random coefficients (2 for each fourrier expansion / batch)
        coeff = np.random.uniform(-1, 1, (self.n_batch, 2 * MaxFourrierExp))

        for n in range(MaxFourrierExp):
            # We use np.newaxis to reshape our coefficients from (n_batch,) to (n_batch, 1). 
            # When multiplied by X (N,), NumPy cleanly broadcasts to (n_batch, N).
            c_cos = (coeff[:, n] / (n + 1))[:, np.newaxis]
            c_sin = (coeff[:, MaxFourrierExp + n] / (n + 1))[:, np.newaxis]

            self.VAL += c_cos * np.cos(2 * np.pi * n * self.X)
            self.VAL += c_sin * np.sin(2 * np.pi * n * self.X)

        # Batch-wise Minimums
        min_vals = np.abs(np.min(self.VAL, axis=1, keepdims=True))
        self.VAL += min_vals + EPS

    def plot(self, max_plots: int = 3):
        """Plots a subset of the batches to avoid visual clutter."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        limit = min(self.n_batch, max_plots)
        for i in range(limit):
            ax.plot(self.X, self.VAL[i], label=f"Batch {i}")
            
        ax.set_title(f"Random Fourier Series (Showing {limit} of {self.n_batch} batches)")
        ax.set_xlabel("X")
        ax.set_ylabel("VAL")
        if self.n_batch > 1:
            ax.legend()
        plt.show()