"""Alpha-policy training against a fine MUSCL Euler reference.

What one training step does, in words:
  1. draw K random shock-forming initial conditions (fresh every epoch),
  2. for each, solve the fine MUSCL reference and the coarse DGSEM trainee from
     the SAME IC -- MUSCL is the fixed target, DGSEM carries the network alpha,
  3. compare the two on a shared uniform cost grid and backprop the cost into
     the network weights.

Everything uses an IMPOSED dt (cfg.dt, no CFL). Gradients flow only through the
DGSEM rollout; the MUSCL reference is a fixed NumPy target.

The reading order below is: the differentiable core (how one rollout's cost and
gradient are computed) -> the data (ICs + references) -> setup helpers ->
one-epoch helpers -> checkpointing -> the train() loop, which reads top-down.

Run from the repo root:
    .venv_spectral/bin/python nn/training/train.py            # full config
    .venv_spectral/bin/python nn/training/train.py --P 6      # order 6
    .venv_spectral/bin/python nn/training/train.py --light    # fast demo
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.solver import rk4_step
from jax_dgsem.indicator import postprocess_alpha
from network.model import save_model, load_model
from network.policy import apply_alpha, build_alpha_model, model_meta
from training.config import TrainConfig
from training.cost import uniform_projector, cost_step, cost_terms
from training.ic import draw_fourier_coeffs, sample_on_dgsem, sample_on_muscl
from muscl.euler import MusclEulerGrid, MusclEulerSolver

PERIODIC = ("periodic", None)


# ===========================================================================
# 1. Differentiable core: the cost of one from-IC DGSEM rollout
# ===========================================================================

def network_alpha(model, U, mesh, cfg):
    """State U -> blending factor alpha from the network.

    Soft post-processing (no hard clip) so gradients stay alive during
    training; the hard clip is only used at deployment. Shape is (n_elem,) for
    the element policy, (n_elem, P) for the nodal/opno policy."""
    raw = apply_alpha(model, U, mesh, cfg.model_type)
    return postprocess_alpha(raw, alpha_max=cfg.alpha_max,
                             diffuse=cfg.alpha_diffuse, hard_clip=False)


def rollout_cost(model, U0, targets, mesh, project, dt, dx_ref, cfg):
    """Roll DGSEM from U0 for len(targets) steps and sum the per-step cost
    against the MUSCL reference. This is the scalar that is differentiated.

    targets: (rollout_steps, 3, N_cost) reference, one row per DGSEM step.
    filter_checkpoint keeps rollout memory ~O(1) in the horizon.

    cfg.bptt_window truncates BACKPROP only: the rollout is chunked into
    windows of that many steps and the carried state is stop_gradient'ed at
    each window boundary, so the forward trajectory (and therefore the loss
    VALUE) is bit-identical to full BPTT while gradients flow through at most
    one window of solver steps. This is what keeps the gradient finite AND
    informative for a policy whose d(alpha)/dU is stiffer than the CNN's: on
    the GNN_PNO branch the full-512-step gradient reached a *finite* 1e28-1e33
    (pure chaos amplification, onset between 128 and 256 steps), which
    clip_by_global_norm turned into unit-norm noise updates."""
    window = cfg.bptt_window or 0
    if window <= 0 or window >= targets.shape[0]:
        n_win, win = 1, targets.shape[0]          # full BPTT (CNN-safe)
    else:
        n_win, win = targets.shape[0] // window, window
    tgt = targets.reshape((n_win, win) + targets.shape[1:]) # -> (n_win, win_size, 3, N_cost )

    @eqx.filter_checkpoint
    def window_cost(U, targets_w):
        def step(U, target):
            alpha = network_alpha(model, U, mesh, cfg)
            U_next = rk4_step(U, alpha, dt, mesh)
            cost = cost_step(project(U_next), target, alpha, dx_ref,
                             cfg.w_osc, cfg.w_acc, cfg.w_alpha)
            return U_next, cost
        U_end, costs = jax.lax.scan(step, U, targets_w) # For each window : apply step to the target, return U_next and cost
        return jax.lax.stop_gradient(U_end), jnp.sum(costs)

    _, costs = jax.lax.scan(window_cost, U0, tgt)
    return jnp.sum(costs)


def rollout_diagnostics(model, U0, targets, mesh, project, dt, dx_ref, cfg):
    """Same rollout (no gradients -> no windowing), but returns the UNWEIGHTED
    (C_osc, C_acc, C_alpha), alpha mean/max, and the per-interface-index mean
    alpha -- the numbers that go into the training-analysis logs.

    The per-interface means (over elements and steps) are the alpha-comb
    detector: a position-keyed collapse like the GNN_PNO one (alpha ~ 1 at the
    SAME interface index of every element, ~0 elsewhere) is invisible in the
    pooled mean but screams in max_j(mean) >> min_j(mean)."""
    P = mesh.P

    @eqx.filter_checkpoint
    def step(U, target):
        alpha = network_alpha(model, U, mesh, cfg)
        U_next = rk4_step(U, alpha, dt, mesh)
        osc, acc, alph = cost_terms(project(U_next), target, alpha, dx_ref)
        iface = (alpha.mean(axis=0) if alpha.ndim == 2
                 else jnp.full((P,), alpha.mean()))
        return U_next, jnp.concatenate(
            [jnp.stack([osc, acc, alph, jnp.mean(alpha), jnp.max(alpha)]),
             iface])

    _, per_step = jax.lax.scan(step, U0, targets)
    osc, acc, alph = per_step[:, 0].sum(), per_step[:, 1].sum(), per_step[:, 2].sum()
    return jnp.concatenate(
        [jnp.stack([osc, acc, alph, per_step[:, 3].mean(), per_step[:, 4].max()]),
         per_step[:, 5:].mean(axis=0)])


def batch_mean(fn, model, starts, targets_batch, *args):
    """Average `fn` over a batch of ICs (vmap). Periodic BCs make the mesh
    identical for every IC, so the vmap is only over (U0, targets)."""
    return jax.vmap(lambda U0, tgt: fn(model, U0, tgt, *args)
                    )(starts, targets_batch)


def make_step_functions(mesh, project, dt, dx_ref, cfg, optimizer):
    """Build the three jitted functions the loop calls: one Adam step, the
    validation loss, and the validation cost breakdown."""
    core = (mesh, project, dt, dx_ref, cfg)
    skip_norm = cfg.grad_skip_norm or 0.0

    def mean_loss(model, starts, targets):
        return jnp.mean(batch_mean(rollout_cost, model, starts, targets, *core))

    @eqx.filter_jit
    def train_step(model, opt_state, starts, targets):
        """One guarded Adam step: the update is SKIPPED outright when the
        global gradient norm is non-finite or exceeds cfg.grad_skip_norm.
        apply_if_finite (in the optimizer) only catches NaN/Inf -- the GNN_PNO
        failure mode was FINITE 1e28-1e33 gradients (BPTT chaos amplification)
        that clip_by_global_norm rescaled into unit-norm noise steps. Healthy
        batch gradients here are O(1-1e2). Returns (model, opt_state, loss,
        gnorm, applied)."""
        loss, grads = eqx.filter_value_and_grad(mean_loss)(model, starts, targets)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
        ok = jnp.isfinite(gnorm)
        if skip_norm > 0:
            ok = ok & (gnorm < skip_norm)

        def apply(_):
            updates, new_state = optimizer.update(grads, opt_state, model)
            return eqx.apply_updates(model, updates), new_state

        def skip(_):
            return model, opt_state

        model_out, opt_out = jax.lax.cond(ok, apply, skip, None)
        return model_out, opt_out, loss, gnorm, ok

    @eqx.filter_jit
    def eval_loss(model, starts, targets):
        # nanmean, NOT mean: the fixed n_val validation set is small, and a
        # not-yet-converged policy occasionally destabilizes ONE of those ICs'
        # rollouts. A plain mean would blank the WHOLE epoch's reported
        # val_loss to NaN from a single bad IC.
        return jnp.nanmean(batch_mean(rollout_cost, model, starts, targets, *core))

    @eqx.filter_jit
    def eval_breakdown(model, starts, targets):
        per_ic = batch_mean(rollout_diagnostics, model, starts, targets, *core)
        n_finite = jnp.sum(jnp.all(jnp.isfinite(per_ic), axis=1))
        return jnp.nanmean(per_ic, axis=0), n_finite

    return train_step, eval_loss, eval_breakdown


# ===========================================================================
# 2. Data: shock-forming ICs and their MUSCL references
# ===========================================================================

def muscl_reference(coeffs, cfg):
    """Fine MUSCL Euler solve from the shared IC, projected onto the cost grid.

    Returns (rollout_steps, 3, N_cost): the reference AFTER each DGSEM step
    (row i is the target for DGSEM step i+1). Pure NumPy -- no gradients."""
    grid = MusclEulerGrid(cfg.N_muscl, cfg.xL, cfg.xR)
    grid.set_state(sample_on_muscl(coeffs, grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps)
    traj = solver.trajectory(cfg.rollout_steps, cfg.muscl_substeps)
    block = cfg.N_muscl // cfg.N_cost
    projected = traj.reshape(traj.shape[0], 3, cfg.N_cost, block).mean(axis=-1)
    return projected[1:]                      # drop the IC snapshot


class EpochData:
    """The K (IC, reference) pairs used for one epoch."""

    def __init__(self, dgsem_ics, refs, coeffs_list):
        self.dgsem_ics = dgsem_ics        # (K, 3, n_elem, Nn)
        self.refs = refs                  # (K, rollout_steps, 3, N_cost)
        self.coeffs_list = coeffs_list    # kept for the snapshot plots

    @property
    def size(self):
        return self.dgsem_ics.shape[0]

    @property
    def batch(self):
        return self.dgsem_ics, self.refs


def draw_one_ic(rng, cfg):
    """One shock-forming IC's Fourier coefficients."""
    return draw_fourier_coeffs(rng, cfg.n_fourier_modes, cfg.ic_amp,
                               cfg.xL, cfg.xR,
                               target_compression=cfg.target_compression,
                               vel_modes=cfg.vel_modes)


def generate_epoch_data(rng, n_ics, mesh, cfg):
    """Draw n_ics ICs and build their references. ICs whose MUSCL reference
    blew up are dropped (should not happen for the bounded ICs)."""
    ics, refs, coeffs_list = [], [], []
    for _ in range(n_ics):
        coeffs = draw_one_ic(rng, cfg)
        ref = muscl_reference(coeffs, cfg)
        if not np.isfinite(ref).all():
            continue
        ics.append(np.asarray(sample_on_dgsem(coeffs, mesh, cfg.xL)))
        refs.append(ref)
        coeffs_list.append(coeffs)
    return EpochData(jnp.asarray(np.stack(ics)), jnp.asarray(np.stack(refs)),
                     coeffs_list)


def sample_batch(rng, data, cfg):
    """A random batch of ICs (with replacement) from the epoch's pool."""
    idx = rng.integers(0, data.size, size=min(cfg.batch_size, data.size))
    return data.dgsem_ics[idx], data.refs[idx]


# ===========================================================================
# 3. Setup helpers (each builds one thing)
# ===========================================================================

def validate_config(cfg):
    assert cfg.muscl_cells_per_elem % cfg.proj_pts_per_elem == 0, \
        "muscl_cells_per_elem must be a multiple of proj_pts_per_elem"
    assert cfg.N_muscl % cfg.N_cost == 0, "N_muscl must be a multiple of N_cost"
    if cfg.bptt_window and 0 < cfg.bptt_window < cfg.rollout_steps:
        assert cfg.rollout_steps % cfg.bptt_window == 0, \
            "bptt_window must divide rollout_steps"


def build_discretization(cfg):
    """The coarse DGSEM mesh, the cost-grid projector, and its spacing."""
    basis = GLLBasis(cfg.P)
    mesh = Mesh1D(basis, cfg.n_elem, cfg.xL, cfg.xR, PERIODIC, PERIODIC)
    project = uniform_projector(cfg.P, cfg.n_elem, cfg.proj_pts_per_elem)
    dx_ref = (cfg.xR - cfg.xL) / cfg.N_cost
    return mesh, project, dx_ref


def build_model(cfg, stable_init=True):
    """stable_init=True gives the saturated stable Stage-2 cold start; pass
    False for a trainable readout when the model will be PP-pretrained."""
    key = jax.random.split(jax.random.PRNGKey(cfg.seed))[1]
    return build_alpha_model(cfg.model_type, cfg.P, cfg.width, cfg.kernel_size,
                             cfg.depth, key, cfg.alpha_init, stable_init,
                             b_res=cfg.bool_res, precondition=cfg.precondition,
                             opno_hidden=cfg.opno_hidden,
                             opno_channels=cfg.opno_channels,
                             fusion_hidden=cfg.fusion_hidden)


def build_optimizer(cfg, model):
    """Adam with two guards: clip gradient spikes (forming shocks), and never
    apply a non-finite update (skip that batch) so one bad rollout cannot
    permanently poison the weights."""

    if not cfg.bool_constant_lr:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=cfg.lr, peak_value=3e-4,   # more conservative than O4
            warmup_steps=200, decay_steps=2000, end_value=1e-6
        )
    else:
        lr_schedule = cfg.lr

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                    optax.adam(learning_rate=lr_schedule)),
        max_consecutive_errors=10 * cfg.batches_per_epoch)
    
    return optimizer, optimizer.init(eqx.filter(model, eqx.is_array))


# ===========================================================================
# 4. One epoch: train on batches, then evaluate on the fixed validation set
# ===========================================================================

def run_training_epoch(model, opt_state, data, train_step, rng, cfg, history,
                       epoch):
    """One guarded Adam step per batch; returns the updated (model, opt_state),
    the mean batch loss, and how many updates the gradient guard skipped."""
    losses, n_skipped = [], 0
    for _ in range(cfg.batches_per_epoch):
        starts, targets = sample_batch(rng, data, cfg)
        model, opt_state, loss, gnorm, ok = train_step(
            model, opt_state, starts, targets)
        losses.append(float(loss))
        n_skipped += int(not bool(ok))
        history.record_batch(epoch, float(loss), float(gnorm))
    return model, opt_state, float(np.nanmean(losses)), n_skipped


def evaluate(model, val_batch, eval_loss, eval_breakdown):
    """Validation loss + unweighted cost breakdown + alpha statistics.

    val_n_finite reports how many of the fixed n_val ICs produced a finite
    rollout this epoch -- < n_val is a signal to look at stability, not a
    training-loop bug by itself. val_alpha_if_max/min are the per-interface-
    index means (the alpha-comb detector, see rollout_diagnostics)."""
    stats, n_finite = eval_breakdown(model, *val_batch)
    stats = np.asarray(stats)
    osc, acc, alph, a_mean, a_max = stats[:5]
    iface = stats[5:]
    return dict(val_loss=float(eval_loss(model, *val_batch)),
                val_c_osc=float(osc), val_c_acc=float(acc),
                val_c_alpha=float(alph), val_alpha_mean=float(a_mean),
                val_alpha_max=float(a_max),
                val_alpha_if_max=float(np.nanmax(iface)),
                val_alpha_if_min=float(np.nanmin(iface)),
                val_n_finite=int(n_finite))


def log_epoch(epoch, train_loss, m, n_skipped, n_val, is_best):
    flag = "  *best*" if is_best else ""
    skip = f"  (skipped {n_skipped} guarded)" if n_skipped else ""
    val_bad = f"  (val {m['val_n_finite']}/{n_val} ICs finite)" \
        if m["val_n_finite"] < n_val else ""
    # comb detector: a position-keyed alpha (same interface index hot in every
    # element) shows as if_max >> if_min while the pooled mean looks tame
    comb = f"  iface[{m['val_alpha_if_min']:.3f}..{m['val_alpha_if_max']:.3f}]"
    print(f"epoch {epoch:3d}: train {train_loss:.6e}   val {m['val_loss']:.6e}   "
          f"[C_osc {m['val_c_osc']:.3e}  C_acc {m['val_c_acc']:.3e}  "
          f"C_alpha {m['val_c_alpha']:.3e}  alpha mean {m['val_alpha_mean']:.3f} "
          f"max {m['val_alpha_max']:.3f}]{comb}{flag}{skip}{val_bad}")


def save_snapshot(model, mesh, coeffs, cfg, epoch):
    """DGSEM-vs-MUSCL snapshot for the epoch (never fatal to training)."""
    if not (cfg.plot_every and epoch % cfg.plot_every == 0):
        return
    try:
        from training.viz_snapshot import plot_snapshot
        plot_snapshot(model, mesh, coeffs, cfg, epoch,
                      os.path.join(cfg.checkpoint_dir, "snapshots"))
    except Exception as e:
        print(f"  snapshot plot failed: {e}")


def render_recap(hist_path, cfg):
    try:
        from training.plots import plot_training_history
        fig = os.path.join(cfg.checkpoint_dir, "training_recap.png")
        plot_training_history(hist_path, fig)
        print(f"training recap figure -> {fig}")
    except Exception as e:
        print(f"recap plot failed: {e}")


# ===========================================================================
# 5. Metric history and checkpointing
# ===========================================================================

class History:
    """Per-epoch and per-batch metrics, serializable to one .npz."""

    _EPOCH = ("epoch", "train_loss", "val_loss", "val_c_osc", "val_c_acc",
              "val_c_alpha", "val_alpha_mean", "val_alpha_max",
              "val_alpha_if_max", "val_alpha_if_min", "val_n_finite")
    _BATCH = ("batch_loss", "batch_epoch", "batch_gnorm")

    def __init__(self, data=None):
        self.data = data or {k: [] for k in self._EPOCH + self._BATCH}

    def record_batch(self, epoch, loss, gnorm=float("nan")):
        self.data["batch_loss"].append(loss)
        self.data["batch_epoch"].append(epoch)
        self.data["batch_gnorm"].append(gnorm)

    def record_epoch(self, epoch, train_loss, metrics):
        self.data["epoch"].append(epoch)
        self.data["train_loss"].append(train_loss)
        for k in self._EPOCH[2:]:
            self.data[k].append(metrics[k])

    @property
    def best_val(self):
        finite = [v for v in self.data["val_loss"] if np.isfinite(v)]
        return min(finite) if finite else float("inf")

    @property
    def next_epoch(self):
        return self.data["epoch"][-1] + 1 if self.data["epoch"] else 0

    def save(self, path, cfg):
        np.savez(path, w_osc=cfg.w_osc, w_acc=cfg.w_acc, w_alpha=cfg.w_alpha,
                 **{k: np.asarray(v) for k, v in self.data.items()})

    @classmethod
    def load(cls, path):
        d = dict(np.load(path))
        # newer keys (gnorm, comb detector, n_finite) are absent from older
        # histories -- backfill with NaN/-1 ("not tracked") of the right length
        n_ep = len(d["epoch"]) if "epoch" in d else 0
        n_b = len(d["batch_loss"]) if "batch_loss" in d else 0
        out = {}
        for k in cls._EPOCH + cls._BATCH:
            if k in d:
                out[k] = list(d[k])
            else:
                n = n_ep if k in cls._EPOCH else n_b
                out[k] = [-1 if k == "val_n_finite" else float("nan")] * n
        return cls(out)


class Checkpointer:
    """Owns the checkpoint-directory paths and does all save/restore."""

    def __init__(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        self.meta = os.path.join(dirpath, "model_meta.json")
        self.last = os.path.join(dirpath, "alpha_model_last.eqx")
        self.best = os.path.join(dirpath, "alpha_model_best.eqx")
        self.opt = os.path.join(dirpath, "opt_state.eqx")
        self.hist = os.path.join(dirpath, "training_history.npz")
        self.rng = os.path.join(dirpath, "rng_state.json")

    def write_meta(self, cfg):
        with open(self.meta, "w") as f:
            json.dump({**model_meta(cfg),
                       "config": {k: v for k, v in vars(cfg).items()
                                  if isinstance(v, (int, float, bool, str))}},
                      f, indent=2)

    def assert_architecture_matches(self, cfg):
        with open(self.meta) as f:
            old = json.load(f)
        arch = {"model_type": cfg.model_type, "P": cfg.P, "width": cfg.width,
                "kernel_size": cfg.kernel_size, "depth": cfg.depth}
        if cfg.model_type == "opno":
            del arch["kernel_size"]          # unused by the opno model
            arch.update(opno_hidden=cfg.opno_hidden,
                        opno_channels=cfg.opno_channels,
                        fusion_hidden=cfg.fusion_hidden)
        mismatch = {k: (old.get(k), v) for k, v in arch.items()
                    if old.get(k) != v}
        if mismatch:
            raise ValueError(f"--resume: checkpoint architecture does not match "
                             f"cfg (old, new): {mismatch}.")

    def restore(self, model, opt_state, rng):
        """Return (model, opt_state, history) restored bit-exactly; also seeds
        `rng` in place so the data stream continues where it stopped."""
        missing = [p for p in (self.meta, self.last, self.opt, self.hist,
                               self.rng) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"--resume requested but checkpoint files are missing: "
                f"{missing}. Start a fresh run (no --resume) instead.")
        model = load_model(self.last, model)
        opt_state = eqx.tree_deserialise_leaves(self.opt, opt_state)
        with open(self.rng) as f:
            rng.bit_generator.state = json.load(f)
        return model, opt_state, History.load(self.hist)

    def save(self, model, opt_state, rng, history, cfg, is_best):
        history.save(self.hist, cfg)
        save_model(self.last, model)
        eqx.tree_serialise_leaves(self.opt, opt_state)
        with open(self.rng, "w") as f:
            json.dump(rng.bit_generator.state, f)
        if is_best:
            save_model(self.best, model)


# ===========================================================================
# 6. The training loop
# ===========================================================================

def train(cfg=None, resume=False):
    cfg = cfg or TrainConfig()
    validate_config(cfg)

    # --- build every piece once -------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    mesh, project, dx_ref = build_discretization(cfg)

    # Stage 1 (optional): PP-imitation warm-start, so Stage-2 rollouts survive
    # the forming shock from step 0. Skipped on resume (the checkpoint already
    # holds the trained-from-warm-start weights). The pretrained model needs a
    # trainable readout (stable_init=False), otherwise the regression can't move.
    pretraining = cfg.pretrain_epochs > 0 and not resume
    model = build_model(cfg, stable_init=not pretraining)
    if pretraining:
        from training.pretrain import pretrain_pp
        model = pretrain_pp(model, cfg, mesh, cfg.seed, cfg.pretrain_epochs)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        save_model(os.path.join(cfg.checkpoint_dir,
                                "alpha_model_pretrained.eqx"), model)

    optimizer, opt_state = build_optimizer(cfg, model)
    train_step, eval_loss, eval_breakdown = make_step_functions(
        mesh, project, cfg.dt, dx_ref, cfg, optimizer)

    # fixed validation set (same every run: its own seed, not checkpointed)
    val_data = generate_epoch_data(np.random.default_rng(cfg.seed + 10_000),
                                   cfg.n_val, mesh, cfg)
    viz_coeffs = val_data.coeffs_list[0]

    # --- resume or start fresh --------------------------------------------
    ckpt = Checkpointer(cfg.checkpoint_dir)
    history = History()
    start_epoch = 0
    if resume:
        ckpt.assert_architecture_matches(cfg)
        model, opt_state, history = ckpt.restore(model, opt_state, rng)
        start_epoch = history.next_epoch
        print(f"resuming from epoch {start_epoch} (best_val {history.best_val:.6e})")
        if start_epoch >= cfg.epochs:
            print("nothing to do (raise --epochs to continue)")
            return model
    ckpt.write_meta(cfg)

    # --- epoch loop --------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        data = generate_epoch_data(rng, cfg.K, mesh, cfg)
        if data.size == 0:
            print(f"epoch {epoch:3d}: all references invalid, resampling")
            continue

        model, opt_state, train_loss, n_skipped = run_training_epoch(
            model, opt_state, data, train_step, rng, cfg, history, epoch)
        metrics = evaluate(model, val_data.batch, eval_loss, eval_breakdown)

        is_best = metrics["val_loss"] < history.best_val
        history.record_epoch(epoch, train_loss, metrics)
        ckpt.save(model, opt_state, rng, history, cfg, is_best)

        log_epoch(epoch, train_loss, metrics, n_skipped, cfg.n_val, is_best)
        save_snapshot(model, mesh, viz_coeffs, cfg, epoch)

    render_recap(ckpt.hist, cfg)
    return model


# ===========================================================================
# 7. Command line
# ===========================================================================

def _config_from_args(args):
    from dataclasses import replace
    if args.light:
        cfg = TrainConfig.light()
    elif args.P is not None:
        # for_order picks a subcell-CFL-stable dt and matching rollout_steps
        cfg = TrainConfig.for_order(args.P, epochs=args.epochs or 100)
    else:
        cfg = TrainConfig()
    overrides = {k: v for k, v in (
        ("n_elem", args.n_elem), ("model_type", args.model_type),
        ("epochs", args.epochs), ("seed", args.seed), ("lr", args.lr),
        ("dt", args.dt), ("rollout_steps", args.rollout_steps),
        ("pretrain_epochs", args.pretrain_epochs),
        ("precondition", args.precondition),
        ("bptt_window", args.bptt_window),
        ("checkpoint_dir", args.checkpoint_dir)) if v is not None}
    return replace(cfg, **overrides) if overrides else cfg


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="fast demonstration training (TrainConfig.light)")
    ap.add_argument("--P", type=int, default=None, help="polynomial order")
    ap.add_argument("--n-elem", type=int, default=None)
    ap.add_argument("--model-type", default=None,
                    choices=["opno", "nodal", "element"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bptt-window", type=int, default=None,
                    help="truncated-BPTT window in solver steps (0 = full "
                         "BPTT); must divide rollout_steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--pretrain-epochs", type=int, default=None,
                    help="Stage-1 PP-imitation warm-start epochs (0 = off)")
    ap.add_argument("--precondition", action="store_true", default=None,
                    help="ZCA-whiten the data-channel feature matrix before the "
                         "one-hot is appended (config.precondition)")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint dir after a kill")
    args = ap.parse_args()

    cfg = _config_from_args(args)
    print(f"config: {cfg}")
    train(cfg, resume=args.resume)
