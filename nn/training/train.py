"""Alpha-policy training against a fine MUSCL Euler reference.

Pipeline (per the rewritten design):
  1. MUSCL_grid  : fine uniform periodic FV grid  (n_elem * muscl_cells_per_elem)
  2. DGSEM_grid  : coarse periodic hybrid-DGSEM mesh (n_elem, P)  -- the trainee
  3. a shared periodic random-Fourier Euler IC is applied to BOTH grids
  4. the fine MUSCL solve is the reference; the coarse DGSEM solve is rolled
     from the SAME IC with the network alpha, and both are compared on a shared
     uniform cost grid via the cost function.

Time stepping uses an IMPOSED dt (cfg.dt) everywhere -- no CFL. MUSCL substeps
dt/muscl_substeps per DGSEM step. Gradients flow only through the DGSEM
rollout (the MUSCL reference is a fixed NumPy target).

Run from the repo root:
    .venv_spectral/bin/python nn/training/train.py            # full config
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
from jax_dgsem.indicator import modal_energy, postprocess_alpha
from network.model import AlphaModel, save_model, load_model
from training.config import TrainConfig
from training.cost import uniform_projector, cost_step, cost_terms
from training.ic import draw_fourier_coeffs, sample_on_dgsem, sample_on_muscl
from muscl.euler import MusclEulerGrid, MusclEulerSolver

PERIODIC = ("periodic", None)


# ---------------------------------------------------------------------------
# Alpha from the network
# ---------------------------------------------------------------------------

def network_alpha(model, U, mesh, cfg: TrainConfig):
    """Modal-energy features -> network -> soft post-processing (no hard clip:
    keeps gradients alive; the hard clip is only applied at deployment)."""
    feats = modal_energy(U, mesh).T          # (Nn, n_elem) channels-first
    raw = model(feats)
    return postprocess_alpha(raw, alpha_max=cfg.alpha_max,
                             diffuse=cfg.alpha_diffuse, hard_clip=False)


# ---------------------------------------------------------------------------
# Differentiable from-IC rollout of the DGSEM trainee
# ---------------------------------------------------------------------------

def rollout_loss(model, U0, targets, mesh, project, dt, dx_ref, cfg):
    """Roll the DGSEM solver from U0 for len(targets) steps with the network
    alpha; accumulate the per-step cost against the MUSCL reference.

    targets: (rollout_steps, 3, N_cost) MUSCL reference already on the cost
    grid, one row per DGSEM step. eqx.filter_checkpoint keeps rollout memory
    ~O(1) in the horizon (recompute-on-backward)."""

    @eqx.filter_checkpoint
    def body(U, tgt):
        alpha = network_alpha(model, U, mesh, cfg)
        U1 = rk4_step(U, alpha, dt, mesh)
        c = cost_step(project(U1), tgt, alpha, dx_ref,
                      cfg.w_osc, cfg.w_acc, cfg.w_alpha)
        return U1, c

    _, cs = jax.lax.scan(body, U0, targets)
    return jnp.sum(cs)


def rollout_terms(model, U0, targets, mesh, project, dt, dx_ref, cfg):
    """Same rollout, unweighted (C_osc, C_acc, C_alpha) + alpha stats, for the
    training-analysis logs."""

    @eqx.filter_checkpoint
    def body(U, tgt):
        alpha = network_alpha(model, U, mesh, cfg)
        U1 = rk4_step(U, alpha, dt, mesh)
        osc, acc, alph = cost_terms(project(U1), tgt, alpha, dx_ref)
        return U1, jnp.stack([osc, acc, alph, jnp.mean(alpha), jnp.max(alpha)])

    _, out = jax.lax.scan(body, U0, targets)
    osc, acc, alph = out[:, 0].sum(), out[:, 1].sum(), out[:, 2].sum()
    return jnp.stack([osc, acc, alph, out[:, 3].mean(), out[:, 4].max()])


def make_train_step(mesh, project, dt, dx_ref, cfg: TrainConfig, optimizer):
    """With periodic BCs the mesh is identical for every IC, so the batch vmap
    is just over (U0, targets) -- no per-IC boundary state to map over."""

    def batch_loss(model, U0s, targets_b):
        losses = jax.vmap(
            lambda U0, tgt: rollout_loss(model, U0, tgt, mesh, project, dt,
                                         dx_ref, cfg))(U0s, targets_b)
        return jnp.mean(losses)

    @eqx.filter_jit
    def train_step(model, opt_state, U0s, targets_b):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(
            model, U0s, targets_b)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, U0s, targets_b):
        return batch_loss(model, U0s, targets_b)

    @eqx.filter_jit
    def eval_breakdown(model, U0s, targets_b):
        terms = jax.vmap(
            lambda U0, tgt: rollout_terms(model, U0, tgt, mesh, project, dt,
                                          dx_ref, cfg))(U0s, targets_b)
        return terms.mean(axis=0)

    return train_step, eval_loss, eval_breakdown


# ---------------------------------------------------------------------------
# MUSCL reference + epoch data
# ---------------------------------------------------------------------------

def muscl_reference_on_cost_grid(coeffs, cfg: TrainConfig):
    """Fine MUSCL Euler solve from the shared IC, projected to the cost grid.

    Returns (rollout_steps, 3, N_cost): the reference AFTER each DGSEM step
    (index i is the target for DGSEM step i+1). NumPy -- no gradients."""
    grid = MusclEulerGrid(cfg.N_muscl, cfg.xL, cfg.xR)
    grid.set_state(sample_on_muscl(coeffs, grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps)
    traj = solver.trajectory(cfg.rollout_steps, cfg.muscl_substeps)
    factor = cfg.N_muscl // cfg.N_cost
    proj = traj.reshape(traj.shape[0], 3, cfg.N_cost, factor).mean(axis=-1)
    return proj[1:]                      # drop the IC snapshot


class EpochData:
    def __init__(self, dgsem_ics, refs_proj, coeffs_list):
        self.dgsem_ics = dgsem_ics       # (K, 3, n_elem, Nn)
        self.refs_proj = refs_proj       # (K, rollout_steps, 3, N_cost)
        self.coeffs_list = coeffs_list   # for visualization


def build_epoch_data(rng: np.random.Generator, n_ics, mesh, cfg: TrainConfig):
    dgsem_ics, refs_proj, coeffs_list = [], [], []
    for _ in range(n_ics):
        coeffs = draw_fourier_coeffs(rng, cfg.n_fourier_modes, cfg.ic_amp,
                                     cfg.xL, cfg.xR,
                                     target_compression=cfg.target_compression,
                                     vel_modes=cfg.vel_modes)
        ref = muscl_reference_on_cost_grid(coeffs, cfg)
        if not np.isfinite(ref).all():
            continue                     # MUSCL blew up (shouldn't for smooth IC)
        dgsem_ics.append(np.asarray(sample_on_dgsem(coeffs, mesh, cfg.xL)))
        refs_proj.append(ref)
        coeffs_list.append(coeffs)
    return EpochData(jnp.asarray(np.stack(dgsem_ics)),
                     jnp.asarray(np.stack(refs_proj)), coeffs_list)


def sample_batch(rng: np.random.Generator, data: EpochData, cfg: TrainConfig):
    K = data.dgsem_ics.shape[0]
    idx = rng.integers(0, K, size=min(cfg.batch_size, K))
    return data.dgsem_ics[idx], data.refs_proj[idx]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig = None, resume: bool = False):
    cfg = cfg or TrainConfig()
    assert cfg.muscl_cells_per_elem % cfg.proj_pts_per_elem == 0, \
        "muscl_cells_per_elem must be a multiple of proj_pts_per_elem"
    assert cfg.N_muscl % cfg.N_cost == 0, "N_muscl must be a multiple of N_cost"

    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    basis = GLLBasis(cfg.P)
    mesh = Mesh1D(basis, cfg.n_elem, cfg.xL, cfg.xR, PERIODIC, PERIODIC)
    project = uniform_projector(cfg.P, cfg.n_elem, cfg.proj_pts_per_elem)
    dx_ref = (cfg.xR - cfg.xL) / cfg.N_cost

    key, k_model = jax.random.split(key)
    model = AlphaModel(cfg.P + 1, cfg.width, cfg.kernel_size, cfg.depth,
                       key=k_model)

    # Guarded optimizer: a forming shock in the rollout can spike or NaN the
    # gradient; clip the spikes and skip (never apply) the non-finite batches
    # so one bad rollout cannot poison the weights permanently.
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                    optax.adam(cfg.lr)),
        max_consecutive_errors=10 * cfg.batches_per_epoch)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step, eval_loss, eval_breakdown = make_train_step(
        mesh, project, cfg.dt, dx_ref, cfg, optimizer)

    # Fixed validation set (generated once).
    val_rng = np.random.default_rng(cfg.seed + 10_000)
    val_data = build_epoch_data(val_rng, cfg.n_val, mesh, cfg)
    val_batch = (val_data.dgsem_ics, val_data.refs_proj)
    viz_coeffs = val_data.coeffs_list[0]      # fixed IC for snapshot plots

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    start_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_c_osc": [], "val_c_acc": [], "val_c_alpha": [],
               "val_alpha_mean": [], "val_alpha_max": [],
               "batch_loss": [], "batch_epoch": []}

    meta_path = os.path.join(cfg.checkpoint_dir, "model_meta.json")
    last_path = os.path.join(cfg.checkpoint_dir, "alpha_model_last.eqx")
    best_path = os.path.join(cfg.checkpoint_dir, "alpha_model_best.eqx")
    opt_path = os.path.join(cfg.checkpoint_dir, "opt_state.eqx")
    hist_path = os.path.join(cfg.checkpoint_dir, "training_history.npz")
    rng_path = os.path.join(cfg.checkpoint_dir, "rng_state.json")

    if resume:
        missing = [p for p in (meta_path, last_path, opt_path, hist_path,
                               rng_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"--resume requested but checkpoint files are missing: "
                f"{missing}. Start a fresh run (no --resume) instead.")
        with open(meta_path) as f:
            old_meta = json.load(f)
        arch = {"in_channels": cfg.P + 1, "width": cfg.width,
                "kernel_size": cfg.kernel_size, "depth": cfg.depth}
        mismatch = {k: (old_meta[k], v) for k, v in arch.items()
                    if old_meta[k] != v}
        if mismatch:
            raise ValueError(
                f"--resume: checkpoint architecture does not match cfg "
                f"(old, new): {mismatch}.")
        model = load_model(last_path, model)
        opt_state = eqx.tree_deserialise_leaves(opt_path, opt_state)
        with open(rng_path) as f:
            rng.bit_generator.state = json.load(f)
        old = dict(np.load(hist_path))
        history = {k: list(old[k]) for k in history}
        start_epoch = int(history["epoch"][-1]) + 1 if history["epoch"] else 0
        best_val = float(np.min(old["val_loss"])) if len(old["val_loss"]) \
            else float("inf")
        print(f"resuming from epoch {start_epoch} (best_val {best_val:.6e})")
        if start_epoch >= cfg.epochs:
            print("nothing to do (raise --epochs to continue)")
            return model

    with open(meta_path, "w") as f:
        json.dump({"in_channels": cfg.P + 1, "width": cfg.width,
                   "kernel_size": cfg.kernel_size, "depth": cfg.depth,
                   "P": cfg.P, "alpha_max": cfg.alpha_max,
                   "config": {k: v for k, v in vars(cfg).items()
                              if isinstance(v, (int, float, bool, str))}},
                  f, indent=2)

    def save_history():
        np.savez(hist_path, w_osc=cfg.w_osc, w_acc=cfg.w_acc,
                 w_alpha=cfg.w_alpha,
                 **{k: np.asarray(v) for k, v in history.items()})

    for epoch in range(start_epoch, cfg.epochs):
        data = build_epoch_data(rng, cfg.K, mesh, cfg)
        if data.dgsem_ics.shape[0] == 0:
            print(f"epoch {epoch:3d}: all references invalid, resampling")
            continue

        losses = []
        for _ in range(cfg.batches_per_epoch):
            batch = sample_batch(rng, data, cfg)
            model, opt_state, loss = train_step(model, opt_state, *batch)
            losses.append(float(loss))
            history["batch_loss"].append(float(loss))
            history["batch_epoch"].append(epoch)

        val = float(eval_loss(model, *val_batch))
        osc, acc, alph, a_mean, a_max = np.asarray(
            eval_breakdown(model, *val_batch))
        n_skipped = int(opt_state.notfinite_count)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.nanmean(losses)))
        history["val_loss"].append(val)
        history["val_c_osc"].append(float(osc))
        history["val_c_acc"].append(float(acc))
        history["val_c_alpha"].append(float(alph))
        history["val_alpha_mean"].append(float(a_mean))
        history["val_alpha_max"].append(float(a_max))
        save_history()

        flag = ""
        if val < best_val:
            best_val = val
            save_model(best_path, model)
            flag = "  *best*"
        save_model(last_path, model)
        eqx.tree_serialise_leaves(opt_path, opt_state)
        with open(rng_path, "w") as f:
            json.dump(rng.bit_generator.state, f)

        skip = f"  (skipped {n_skipped} non-finite)" if n_skipped else ""
        print(f"epoch {epoch:3d}: train {np.nanmean(losses):.6e}   "
              f"val {val:.6e}   [C_osc {osc:.3e}  C_acc {acc:.3e}  "
              f"C_alpha {alph:.3e}  alpha mean {a_mean:.3f} max {a_max:.3f}]"
              f"{flag}{skip}")

        # During-training visualization: DGSEM (current model) vs MUSCL ref.
        if cfg.plot_every and epoch % cfg.plot_every == 0:
            try:
                from training.viz_snapshot import plot_snapshot
                plot_snapshot(model, mesh, viz_coeffs, cfg, epoch,
                              os.path.join(cfg.checkpoint_dir, "snapshots"))
            except Exception as e:
                print(f"  snapshot plot failed: {e}")

    try:
        from training.plots import plot_training_history
        fig = os.path.join(cfg.checkpoint_dir, "training_recap.png")
        plot_training_history(hist_path, fig)
        print(f"training recap figure -> {fig}")
    except Exception as e:
        print(f"recap plot failed: {e}")

    return model


if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="fast demonstration training (TrainConfig.light)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint dir after a kill")
    args = ap.parse_args()

    cfg = TrainConfig.light() if args.light else TrainConfig()
    overrides = {k: v for k, v in (
        ("epochs", args.epochs), ("seed", args.seed), ("lr", args.lr),
        ("dt", args.dt), ("rollout_steps", args.rollout_steps),
        ("checkpoint_dir", args.checkpoint_dir)) if v is not None}
    if overrides:
        cfg = replace(cfg, **overrides)
    print(f"config: {cfg}")
    train(cfg, resume=args.resume)
