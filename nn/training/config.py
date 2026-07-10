"""Hyperparameters for the alpha-policy training (Algorithm 1, Bois et al.)."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # --- discretization (coarse mesh = the mesh the network trains on) -----
    P: int = 5                # polynomial order (matches C++ default)
    n_elem: int = 50          # coarse elements
    xL: float = 0.0
    xR: float = 1.0
    refine: int = 4           # reference mesh = refine * n_elem elements,
                              # advanced with dt/refine, sampled every refine
    cfl: float = 0.3          # per-IC dt = cfl_time_step(U0) on the coarse mesh

    # --- trajectories (paper Sect. 2.3 / Algorithm 1) -----------------------
    n_steps: int = 400        # N: reference trajectory length (coarse steps)
    m: int = 32               # sub-trajectory length the gradient flows through

    # --- alpha post-processing during training (soft: no hard clipping) ----
    alpha_max: float = 1.0    # cap (reference_data.job uses 1.0; C++ CLI default 0.5)
    alpha_diffuse: bool = True

    # --- cost weights (Sect. 3.2; C_vis -> L2 penalty on alpha) ------------
    w_osc: float = 1e-5
    w_acc: float = 1.0
    w_alpha: float = 1e-5

    proj_pts_per_elem: int = 12   # uniform cost-grid points per coarse element

    # --- optimization -------------------------------------------------------
    epochs: int = 30
    K: int = 8                # initial conditions per epoch
    batches_per_epoch: int = 20
    batch_size: int = 16      # I: sub-trajectories per batch
    lr: float = 1e-3
    grad_clip: float = 1.0    # global-norm clip (gradient spikes near the
                              # stability boundary; see paper Sect. 2.3)
    seed: int = 0

    # --- initial conditions -------------------------------------------------
    n_fourier_modes: int = 20
    ic_eps: float = 0.1       # positivity shift for rho and p (paper Sect 3.4)
    shock_ic_fraction: float = 0.5   # fraction of epoch ICs drawn as random
                                     # Riemann problems: shocks inside the m
                                     # window punish an alpha->0 policy, so
                                     # the incentive to collapse disappears

    # --- model ---------------------------------------------------------------
    width: int = 16
    kernel_size: int = 3
    depth: int = 1

    # --- bookkeeping ---------------------------------------------------------
    n_val: int = 4            # validation ICs (fixed at start of training)
    checkpoint_dir: str = "nn/training/checkpoints"

    @classmethod
    def light(cls) -> "TrainConfig":
        """Fast demonstration training (minutes on a laptop CPU).

        Deliberately small and shock-heavy: the policy overfits to shock-tube
        dynamics, which is exactly what the benchmark comparison shows. Good
        enough to land near Persson-Peraire; not a production training.
        """
        return cls(n_elem=32, refine=2, n_steps=200, m=24,
                   epochs=10, K=4, batches_per_epoch=10, batch_size=8,
                   n_val=2, shock_ic_fraction=0.75,
                   checkpoint_dir="nn/training/checkpoints_light")
