"""Hyperparameters for the alpha-policy training.

Training setup (rewritten): a fine MUSCL Euler solve on MUSCL_grid is the
reference; a coarse hybrid-DGSEM solve on DGSEM_grid is the trainee. The same
periodic random-Fourier IC is applied to both grids, both are advanced with an
IMPOSED dt (no CFL), and the network alpha is trained so the DGSEM propagation
matches the MUSCL reference under the cost function.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- DGSEM_grid (coarse mesh = the mesh the network trains on) ----------
    P: int = 4                # polynomial order (matches C++ default)
    n_elem: int = 32          # coarse DGSEM elements
    xL: float = 0.0
    xR: float = 1.0

    # --- MUSCL_grid (fine reference grid) ----------------------------------
    muscl_cells_per_elem: int = 64   # MUSCL_grid has n_elem * this cells;
                                     # must be a multiple of proj_pts_per_elem
    muscl_substeps: int = 4          # MUSCL substeps per DGSEM step (dt/this)

    # --- time stepping (IMPOSED dt, no CFL) --------------------------------
    dt: float = 2e-4          # DGSEM coarse step; MUSCL step = dt/muscl_substeps
    rollout_steps: int = 1024  # DGSEM steps rolled from the IC and compared to
                              # the MUSCL reference (the training horizon)

    # --- alpha post-processing during training (soft: no hard clipping) ----
    alpha_max: float = 1.0    # cap (C++ CLI default 0.5)
    alpha_diffuse: bool = False

    # --- cost weights (C_vis -> L2 penalty on alpha) -----------------------
    w_osc: float = 1e-5
    w_acc: float = 1.0
    w_alpha: float = 1e-5

    proj_pts_per_elem: int = 16   # uniform cost-grid points per coarse element

    # --- optimization -------------------------------------------------------
    epochs: int = 30
    K: int = 16               # initial conditions generated per epoch
    batches_per_epoch: int = 20
    batch_size: int = 8       # ICs per gradient step (from-IC rollouts)
    lr: float = 1e-3
    grad_clip: float = 1.0    # global-norm clip (gradient spikes when a shock
                              # forms in the rollout)
    seed: int = 0

    # --- initial conditions -------------------------------------------------
    n_fourier_modes: int = 12
    ic_amp: float = 0.4        # half-amplitude of rho/p about 1.0 (must be <1
                               # for positivity). Bounded away from vacuum so
                               # the imposed dt stays CFL-stable.
    vel_modes: int = 3         # velocity Fourier modes (few, low-wavenumber ->
                               # broad compression -> reliable shock every draw)
    shock_time_fraction: float = 0.5   # velocity is scaled so a discontinuity
                               # develops at ~this fraction of the rollout
                               # (fully formed well before the final step);
                               # None -> no shock control (amplitude-only vel).

    # --- model ---------------------------------------------------------------
    width: int = 16
    kernel_size: int = 3
    depth: int = 1

    # --- bookkeeping ---------------------------------------------------------
    n_val: int = 4            # validation ICs (fixed at start of training)
    plot_every: int = 1       # save a DGSEM-vs-MUSCL snapshot every N epochs
                              # (0 disables during-training visualization)
    checkpoint_dir: str = "nn/training/checkpoints"

    @property
    def N_muscl(self) -> int:
        return self.n_elem * self.muscl_cells_per_elem

    @property
    def N_cost(self) -> int:
        return self.n_elem * self.proj_pts_per_elem

    @property
    def target_compression(self):
        """Velocity compression C for the shock-forming IC (None if disabled).

        Forcing a shock into a very short rollout needs an extreme (CFL-
        breaking) compression, so this returns None when there are too few
        steps for the requested fraction to be physical."""
        if self.shock_time_fraction is None:
            return None
        from training.ic import compression_for_shock_fraction
        return compression_for_shock_fraction(
            self.shock_time_fraction, self.rollout_steps, self.dt)

    @classmethod
    def light(cls) -> "TrainConfig":
        """Fast demonstration training (minutes on a laptop CPU).

        Rollout too short to form a shock without breaking the imposed-dt CFL,
        so shock control is off here -- this is a pipeline smoke test."""
        return cls(n_elem=32, muscl_cells_per_elem=16, muscl_substeps=2,
                   dt=2e-4, rollout_steps=64, proj_pts_per_elem=8,
                   n_fourier_modes=8, ic_amp=0.4, shock_time_fraction=None,
                   epochs=10, K=4, batches_per_epoch=10, batch_size=4,
                   n_val=2, checkpoint_dir="nn/training/checkpoints_light")
