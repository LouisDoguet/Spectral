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
    P: int = 6                # polynomial order (matches C++ default)
    n_elem: int = 64          # coarse DGSEM elements
    xL: float = 0.0
    xR: float = 1.0

    # --- MUSCL_grid (fine reference grid) ----------------------------------
    muscl_cells_per_elem: int = 64   # MUSCL_grid has n_elem * this cells;
                                     # must be a multiple of proj_pts_per_elem
    muscl_substeps: int = 4          # MUSCL substeps per DGSEM step (dt/this)

    # --- time stepping (IMPOSED dt, no CFL) --------------------------------
    dt: float = 2e-4          # DGSEM coarse step; MUSCL step = dt/muscl_substeps
    rollout_steps: int = 512  # DGSEM steps rolled from the IC and compared to
                              # the MUSCL reference (the training horizon)

    # --- alpha post-processing during training (soft: no hard clipping) ----
    alpha_max: float = 1.0    # cap (C++ CLI default 0.5)
    alpha_diffuse: bool = False

    # --- cost weights (C_vis -> L2 penalty on alpha) -----------------------
    w_osc: float = 1.0
    w_acc: float = 1.0
    w_alpha: float = 1.0

    proj_pts_per_elem: int = 64   # uniform cost-grid points per coarse element

    # --- optimization -------------------------------------------------------
    epochs: int = 1000
    K: int = 48               # initial conditions generated per epoch
    batches_per_epoch: int = 20
    batch_size: int = 8       # ICs per gradient step (from-IC rollouts)
    lr: float = 3e-4
    bool_constant_lr: bool = False
                              # Constant learning rate or warm up, with initial value = lr
                              # Max value = 3e-4 (Adam optim. reference value)
    grad_clip: float = 1.0    # global-norm clip (gradient spikes when a shock
                              # forms in the rollout)
    grad_skip_norm: float = 1e3
                              # SKIP (don't apply) any update whose global
                              # gradient norm exceeds this. apply_if_finite
                              # only catches NaN/Inf; the GNN_PNO failure mode
                              # was FINITE gradients of norm 1e28-1e33 (BPTT
                              # chaos amplification) that clip_by_global_norm
                              # happily rescaled into unit-norm noise steps.
                              # The healthy CNN batch gradients are O(1-1e2),
                              # so 1e3 rejects only pathological batches.
                              # 0 or None disables the guard.
    bptt_window: int = 32     # truncated-BPTT window: gradients flow through
                              # at most this many consecutive solver steps
                              # (stop_gradient on the carried state between
                              # windows; forward rollout unchanged). Measured
                              # on GNN_PNO: gradient explosion onsets between
                              # 128 and 256 steps once the shock forms. Must
                              # divide rollout_steps; 0 or None = full BPTT
                              # (the CNN tolerates full 512; default is the
                              # safe window for the stiffer opno model).
    seed: int = 5

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
    model_type: str = "opno"   # "opno": element-GNN -> OPNO sensor (P- and
                               #   n_elem-independent), input = [asinh residual
                               #   channels per node, modal spectrum per
                               #   element], output (n_elem, P) per interior
                               #   subcell interface.
                               # "nodal": per interior subcell interface CNN,
                               #   input = [data channels, one-hot node
                               #   position] over the node sequence, output
                               #   (n_elem, P).
                               # "element": legacy per-element modal-energy CNN,
                               #   output (n_elem,).
    alpha_init: float = 1e-3   # untrained-policy alpha (used when NOT
                               # pretraining). Must be high enough that the
                               # coarse scheme survives the FORMING shock from
                               # step 0 (alpha~0.05 blows up); with pretrain_pp
                               # the PP warm-start provides the stable start
                               # instead, so alpha_init is irrelevant.
    pretrain_epochs: int = 1000 # Stage-1 PP-imitation warm-start (0 = off).
                               # Regresses the network toward the Persson-
                               # Peraire indicator on PP-driven rollout states,
                               # so Stage-2 rollouts survive the forming shock
                               # from step 0. A warm start, not a target --
                               # and deliberately CAPPED: on GNN_PNO, driving
                               # the imitation to MSE ~3e-3 (4000 epochs)
                               # inflated the layer spectral norms to 6-16 (the
                               # working CNN sits at <5) and with them the
                               # closed-loop Lipschitz constant. See
                               # pretrain_target_mse.
    pretrain_lr: float = 3e-4
    pretrain_target_mse: float = 1e-2
                               # early-stop the imitation once the epoch MSE
                               # falls below this (the ~200-epoch quality that
                               # produced the only healthy GNN_PNO run).
    width: int = 64            # CNN width ("nodal"/"element") or element-GNN
                               #   latent width ("opno")
    bool_res:bool = True       # Boolean to activate a residual block, or a simple 2 layers CNN
                               #   ("nodal"/"element" only)
    kernel_size: int = 3       # ("nodal"/"element" only; the opno model has no
                               #   convolution)
    depth: int = 1             # ResBlock count ("nodal"/"element") or element-
                               #   GNN message-passing rounds = neighbourhood
                               #   radius in elements ("opno")
    precondition: bool = True  # ZCA-whiten the (N_points, N_features) data-channel
                               #   matrix before the one-hot is appended and fed to
                               #   the network (per-state instance whitening).
                               #   "nodal"/"element" only: the opno model's
                               #   preconditioning is formula-based (asinh
                               #   residual scales + clipped log energy).
    # opno-model sizes (P-independent: none of these depends on P)
    opno_hidden: int = 64      # hidden width of every shared token MLP
    opno_channels: int = 64     # OPNO mode/nodal feature channels (C)
    fusion_hidden: int = 64    # width of the pointwise fuse layer

    # --- bookkeeping ---------------------------------------------------------
    n_val: int = 16            # validation ICs (fixed at start of training)
    plot_every: int = 25       # save a DGSEM-vs-MUSCL snapshot every N epochs
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

    @staticmethod
    def stable_dt(P, n_elem=32, xL=0.0, xR=1.0, lambda_ref=8.0, cfl_fv=1.8,
                  dt_cap=2e-4):
        """Largest imposed dt that keeps the SUBCELL FV part stable.

        The blend can push a subcell to pure first-order FV (alpha=1), and the
        FV CFL is set by the smallest GLL subcell, width ~ dx/(P(P+1)), which
        shrinks fast with P. Calibrated: cfl_fv~1.8 is stable, ~3.7 blows up.
        Capped at dt_cap so low orders keep the validated 2e-4."""
        dx = (xR - xL) / n_elem
        return min(dt_cap, cfl_fv * dx / (P * (P + 1)) / lambda_ref)

    @classmethod
    def for_order(cls, P, target_T=0.1024, epochs=100, **kw):
        """Config for a given polynomial order: a subcell-CFL-stable dt, and
        rollout_steps scaled to keep the physical horizon target_T fixed, so
        the shock still forms at ~shock_time_fraction of the rollout."""
        n_elem = kw.get("n_elem", cls.n_elem)
        dt = cls.stable_dt(P, n_elem)
        rollout = max(1, round(target_T / dt))
        return cls(P=P, dt=dt, rollout_steps=rollout, epochs=epochs,
                   checkpoint_dir=f"nn/training/checkpoints_P{P}", **kw)

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
