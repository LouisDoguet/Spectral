import numpy as np

EPS = 1e-12

class NN_MUSCL_Solver:
    def __init__(self, mesh, flux_fn, speed_fn, dt: float = None):
        """
        Initializes the generalized MUSCL solver.
        
        Args:
            mesh: Instance of NN_Mesh containing X and VAL.
            flux_fn: A function that returns the flux F(u).
            speed_fn: A function that returns the local wave speed F'(u).
            cfl: Courant-Friedrichs-Lewy number.
        """

        self.mesh = mesh
        self.flux_fn = flux_fn
        self.speed_fn = speed_fn
        self.dt = dt
        self.dx = self.mesh.X[1] - self.mesh.X[0]
        
    def minmod(self, r):
        """Minmod slope limiter."""
        return np.maximum(0.0, np.minimum(1.0, r))

    def compute_rhs(self, u):
        """
        Computes the spatial derivative using MUSCL reconstruction 
        and the Rusanov (Local Lax-Friedrichs) flux.
        """
        # Forward and backward differences
        du_fwd = np.roll(u, -1) - u           # u_{i+1} - u_i
        du_bwd = u - np.roll(u, 1)            # u_i - u_{i-1}
        
        # --- 1. Reconstruct Left State (u_L) at interface i+1/2 ---
        r_i = du_bwd / (du_fwd + EPS)
        phi_i = self.minmod(r_i)
        u_L = u + 0.5 * phi_i * du_fwd
        
        # --- 2. Reconstruct Right State (u_R) at interface i+1/2 ---
        # The right state comes from cell i+1 looking backward
        du_fwd_next = np.roll(du_fwd, -1)     # u_{i+2} - u_{i+1}
        r_ip1 = du_fwd / (du_fwd_next + EPS)
        phi_ip1 = self.minmod(r_ip1)
        u_R = np.roll(u, -1) - 0.5 * phi_ip1 * du_fwd_next
        
        # --- 3. Compute Rusanov (Local Lax-Friedrichs) Flux ---
        F_L = self.flux_fn(u_L)
        F_R = self.flux_fn(u_R)
        
        a_L = self.speed_fn(u_L)
        a_R = self.speed_fn(u_R)
        
        # Local maximum wave speed at the interface
        a_max = np.maximum(np.abs(a_L), np.abs(a_R))
        
        # Rusanov flux formula
        flux = 0.5 * (F_L + F_R) - 0.5 * a_max * (u_R - u_L)
        
        # Net flux difference: F_{i+1/2} - F_{i-1/2}
        flux_diff = flux - np.roll(flux, 1)
        
        # Return the Right Hand Side: - dF/dx
        return -flux_diff / self.dx

    def step(self):
        """RK2 Time Step."""
        u_n = self.mesh.VAL.copy()
        
        # RK2 Stage 1
        rhs_n = self.compute_rhs(u_n)
        u_1 = u_n + self.dt * rhs_n

        # RK2 Stage 2
        rhs_1 = self.compute_rhs(u_1)
        self.mesh.VAL = 0.5 * u_n + 0.5 * (u_1 + self.dt * rhs_1)
        
        return self.dt

    def solve(self, t_final: float):
        t = 0.0
        step_count = 0
        while t < t_final:
            dt = self.step()
            if t + dt > t_final:
                dt = t_final - t
            t += dt
            step_count += 1
        print(f"Integration complete. Reached t={t_final:.4f} in {step_count} steps.")