"""Algorithm 1 (Bois et al. 2024) for the hybrid-DGSEM alpha policy.

Each epoch:
  1. draw K random Fourier initial conditions (Sect. 3.4, on primitives),
  2. compute reference trajectories with the Persson-Peraire hybrid solver on
     a refine-x finer mesh (targets only -- no gradients flow through them),
  3. for each batch, sample I sub-trajectories (k, n), start the differentiable
     coarse solver from the restricted reference state S_ref^n(U0_k), roll m
     steps with the network alpha, and do one Adam step on
        J = sum_batch sum_{i=1..m} C(U^i, U_ref^{n+i}).

Run from the repo root:
    .venv_spectral/bin/python nn/training/train.py
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
from jax_dgsem.solver import rk4_step, cfl_time_step
from jax_dgsem.indicator import modal_energy, postprocess_alpha
from network.model import AlphaModel, save_model, load_model
from training.config import TrainConfig
from training.cost import uniform_projector, cost_step, cost_terms
from training.data_loader import (
    random_fourier_ic, random_riemann_ic, wall_states_from_ic, refine_ic,
    restriction_operator, restrict, generate_reference_trajectory)


# ---------------------------------------------------------------------------
# Differentiable rollout
# ---------------------------------------------------------------------------

def network_alpha(model, U, mesh, cfg: TrainConfig):
    """Features -> network -> soft post-processing (no hard clip: keeps
    gradients alive; the hard clip is only applied at deployment)."""
    feats = modal_energy(U, mesh).T          # (Nn, n_elem) channels-first
    raw = model(feats)
    return postprocess_alpha(raw, alpha_max=cfg.alpha_max,
                             diffuse=cfg.alpha_diffuse, hard_clip=False)


def rollout_loss(model, U0, targets, dt, mesh, project, dx_ref,
                 cfg: TrainConfig):
    """Sum of per-step costs over one m-step sub-trajectory.

    targets: (m, 3, n_pts) reference already projected on the cost grid.
    """

    @eqx.filter_checkpoint
    def body(U, tgt):
        alpha = network_alpha(model, U, mesh, cfg)
        U1 = rk4_step(U, alpha, dt, mesh)
        c = cost_step(project(U1), tgt, alpha, dx_ref,
                      cfg.w_osc, cfg.w_acc, cfg.w_alpha)
        return U1, c

    _, cs = jax.lax.scan(body, U0, targets)
    return jnp.sum(cs)


def rollout_terms(model, U0, targets, dt, mesh, project, dx_ref,
                  cfg: TrainConfig):
    """Same rollout, but returns the summed unweighted cost terms and alpha
    statistics — the per-epoch training diagnostics. Also checkpointed: it is
    only used with eqx.filter_jit (no grad), but sharing the same body shape
    keeps memory behavior identical to rollout_loss for large m."""

    @eqx.filter_checkpoint
    def body(U, tgt):
        alpha = network_alpha(model, U, mesh, cfg)
        U1 = rk4_step(U, alpha, dt, mesh)
        osc, acc, alph = cost_terms(project(U1), tgt, alpha, dx_ref)
        return U1, jnp.stack([osc, acc, alph, jnp.mean(alpha),
                              jnp.max(alpha)])

    _, out = jax.lax.scan(body, U0, targets)
    osc, acc, alph = out[:, 0].sum(), out[:, 1].sum(), out[:, 2].sum()
    return jnp.stack([osc, acc, alph, out[:, 3].mean(), out[:, 4].max()])


def _mesh_in_axes(mesh):
    """vmap in_axes pytree: batch only the boundary ghost states."""
    ax = jax.tree_util.tree_map(lambda _: None, mesh)
    return eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state), ax,
                       (0, 0), is_leaf=lambda x: x is None)


def make_train_step(mesh, project, dx_ref, cfg: TrainConfig, optimizer):
    mesh_axes = _mesh_in_axes(mesh)

    def batch_loss(model, starts, targets, dts, bcL, bcR):
        mesh_b = eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state),
                             mesh, (bcL, bcR))
        losses = jax.vmap(
            lambda U0, tgt, dt, msh: rollout_loss(
                model, U0, tgt, dt, msh, project, dx_ref, cfg),
            in_axes=(0, 0, 0, mesh_axes))(starts, targets, dts, mesh_b)
        return jnp.mean(losses)

    @eqx.filter_jit
    def train_step(model, opt_state, starts, targets, dts, bcL, bcR):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(
            model, starts, targets, dts, bcL, bcR)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, starts, targets, dts, bcL, bcR):
        return batch_loss(model, starts, targets, dts, bcL, bcR)

    @eqx.filter_jit
    def eval_breakdown(model, starts, targets, dts, bcL, bcR):
        """(C_osc, C_acc, C_alpha, alpha_mean, alpha_max) averaged over the
        batch — unweighted terms, for the training-analysis plots."""
        mesh_b = eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state),
                             mesh, (bcL, bcR))
        terms = jax.vmap(
            lambda U0, tgt, dt, msh: rollout_terms(
                model, U0, tgt, dt, msh, project, dx_ref, cfg),
            in_axes=(0, 0, 0, mesh_axes))(starts, targets, dts, mesh_b)
        return terms.mean(axis=0)

    return train_step, eval_loss, eval_breakdown


# ---------------------------------------------------------------------------
# Epoch data (reference trajectories + sub-trajectory sampling)
# ---------------------------------------------------------------------------

class EpochData:
    """References for one epoch: fine states, projected targets, per-IC dt/BCs."""

    def __init__(self, refs_fine, refs_proj, dts, bcLs, bcRs):
        self.refs_fine = refs_fine    # list of (n_steps+1, 3, n_f, Nn)
        self.refs_proj = refs_proj    # list of (n_steps+1, 3, n_pts)
        self.dts = dts
        self.bcLs = bcLs
        self.bcRs = bcRs


def build_epoch_data(key, n_ics, mesh_c, mesh_f, project_f,
                     cfg: TrainConfig):
    refs_fine, refs_proj, dts, bcLs, bcRs = [], [], [], [], []
    gen_ref = eqx.filter_jit(generate_reference_trajectory)
    for k in jax.random.split(key, n_ics):
        k_type, k_ic = jax.random.split(k)
        if bool(jax.random.bernoulli(k_type, cfg.shock_ic_fraction)):
            U0 = random_riemann_ic(k_ic, mesh_c, cfg.xL)
        else:
            U0 = random_fourier_ic(k_ic, mesh_c, cfg.xL, cfg.n_fourier_modes,
                                   cfg.ic_eps)
        bcL, bcR = wall_states_from_ic(U0)
        dt = float(cfl_time_step(U0, mesh_c, cfg.cfl))
        mesh_f_k = eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state),
                               mesh_f, (bcL, bcR))
        U0_f = refine_ic(U0, cfg.P, cfg.refine)
        # dt as an array so filter_jit traces it (a Python float would be a
        # static arg -> one recompilation per initial condition)
        ref = gen_ref(U0_f, mesh_f_k, cfg.n_steps, jnp.asarray(dt), cfg.refine,
                      cfg.alpha_max)
        if not bool(jnp.isfinite(ref[-1]).all()):
            continue    # blown-up reference: drop this IC
        refs_fine.append(ref)
        refs_proj.append(jax.vmap(project_f)(ref))
        dts.append(dt)
        bcLs.append(bcL)
        bcRs.append(bcR)
    return EpochData(refs_fine, refs_proj, jnp.asarray(dts),
                     jnp.stack(bcLs), jnp.stack(bcRs))


def sample_batch(rng: np.random.Generator, data: EpochData, B_restr,
                 cfg: TrainConfig):
    """Random set of (k, n) sub-trajectories -> stacked batch arrays."""
    K = len(data.refs_fine)
    ks = rng.integers(0, K, size=cfg.batch_size)
    ns = rng.integers(0, cfg.n_steps - cfg.m + 1, size=cfg.batch_size)
    starts, targets = [], []
    for k, n in zip(ks, ns):
        starts.append(restrict(data.refs_fine[k][n], B_restr, cfg.n_elem))
        targets.append(data.refs_proj[k][n + 1:n + 1 + cfg.m])
    return (jnp.stack(starts), jnp.stack(targets), data.dts[ks],
            data.bcLs[ks], data.bcRs[ks])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig = None, resume: bool = False):
    cfg = cfg or TrainConfig()
    print(cfg.proj_pts_per_elem)
    print(cfg.refine)
    assert cfg.proj_pts_per_elem % cfg.refine == 0, \
        "proj_pts_per_elem must be divisible by refine (shared cost grid)"

    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    basis = GLLBasis(cfg.P)
    dummy = ("wall", np.array([1.0, 0.0, 2.5]))
    mesh_c = Mesh1D(basis, cfg.n_elem, cfg.xL, cfg.xR, dummy, dummy)
    mesh_f = Mesh1D(basis, cfg.n_elem * cfg.refine, cfg.xL, cfg.xR,
                    dummy, dummy)

    project_c = uniform_projector(cfg.P, cfg.n_elem, cfg.proj_pts_per_elem)
    project_f = uniform_projector(cfg.P, cfg.n_elem * cfg.refine,
                                  cfg.proj_pts_per_elem // cfg.refine)
    n_pts = cfg.n_elem * cfg.proj_pts_per_elem
    dx_ref = (cfg.xR - cfg.xL) / n_pts
    B_restr = restriction_operator(cfg.P, cfg.refine)

    key, k_model = jax.random.split(key)
    model = AlphaModel(cfg.P + 1, cfg.width, cfg.kernel_size, cfg.depth,
                       key=k_model)

    # Guarded optimizer: gradients through many RK4 steps spike near the
    # stability boundary, and one non-finite update would poison the weights
    # permanently. Clip the spikes, skip the non-finite batches.
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                    optax.adam(cfg.lr)),
        max_consecutive_errors=10 * cfg.batches_per_epoch)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step, eval_loss, eval_breakdown = make_train_step(
        mesh_c, project_c, dx_ref, cfg, optimizer)

    # Fixed validation set (paper: generated once at the start of training).
    key, k_val = jax.random.split(key)
    val_data = build_epoch_data(k_val, cfg.n_val, mesh_c, mesh_f, project_f,
                                cfg)
    val_batch = sample_batch(np.random.default_rng(12345), val_data, B_restr,
                             cfg)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    start_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_c_osc": [], "val_c_acc": [], "val_c_alpha": [],
               "val_alpha_mean": [], "val_alpha_max": [],
               "batch_loss": [], "batch_epoch": []}

    meta_path = os.path.join(cfg.checkpoint_dir, "model_meta.json")
    last_path = os.path.join(cfg.checkpoint_dir, "alpha_model_last.eqx")
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
                f"(old, new): {mismatch}. Use a different --checkpoint-dir "
                f"for a differently-shaped model.")
        model = load_model(last_path, model)
        opt_state = eqx.tree_deserialise_leaves(opt_path, opt_state)
        with open(rng_path) as f:
            rng.bit_generator.state = json.load(f)
        old = dict(np.load(hist_path))
        history = {k: list(old[k]) for k in history}
        start_epoch = int(history["epoch"][-1]) + 1 if history["epoch"] else 0
        best_val = float(np.min(old["val_loss"])) if len(old["val_loss"]) \
            else float("inf")
        # Replay the per-epoch key splits so the RNG stream from here on
        # matches what an uninterrupted run would have produced (only one
        # split happens per epoch, in the loop below).
        for _ in range(start_epoch):
            key, _ = jax.random.split(key)
        print(f"resuming from epoch {start_epoch} "
              f"(best_val so far: {best_val:.6e})")
        if start_epoch >= cfg.epochs:
            print(f"cfg.epochs={cfg.epochs} <= start_epoch={start_epoch}: "
                 f"nothing to do (raise --epochs to continue training)")
            return model

    # Model hyperparameters, so evaluation scripts can rebuild the template
    # for eqx.tree_deserialise_leaves without touching the training config.
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
        key, k_epoch = jax.random.split(key)
        data = build_epoch_data(k_epoch, cfg.K, mesh_c, mesh_f, project_f,
                                cfg)
        if not data.refs_fine:
            print(f"epoch {epoch:3d}: all references blew up, resampling")
            continue

        losses = []
        for _ in range(cfg.batches_per_epoch):
            batch = sample_batch(rng, data, B_restr, cfg)
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
        save_history()   # every epoch: partial runs keep their data

        flag = ""
        if val < best_val:
            best_val = val
            save_model(os.path.join(cfg.checkpoint_dir, "alpha_model_best.eqx"),
                       model)
            flag = "  *best*"
        save_model(last_path, model)
        # opt_state (Adam moments, clip/finite-guard counters) is saved too,
        # so --resume continues optimization smoothly instead of restarting
        # Adam's momentum from scratch (which would show up as a loss spike).
        eqx.tree_serialise_leaves(opt_path, opt_state)
        with open(rng_path, "w") as f:
            json.dump(rng.bit_generator.state, f)
        skip_note = f"  (skipped {n_skipped} non-finite batches so far)" \
            if n_skipped else ""
        print(f"epoch {epoch:3d}: train {np.nanmean(losses):.6e}   "
              f"val {val:.6e}   [C_osc {osc:.3e}  C_acc {acc:.3e}  "
              f"C_alpha {alph:.3e}  alpha mean {a_mean:.3f} max {a_max:.3f}]"
              f"{flag}{skip_note}")

    try:
        from training.plots import plot_training_history
        fig_path = os.path.join(cfg.checkpoint_dir, "training_recap.png")
        plot_training_history(
            os.path.join(cfg.checkpoint_dir, "training_history.npz"), fig_path)
        print(f"training recap figure -> {fig_path}")
    except Exception as e:      # plotting must never kill a finished training
        print(f"recap plot failed: {e}")

    return model


if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="fast demonstration training (TrainConfig.light): "
                         "small mesh, shock-heavy ICs, minutes on a CPU")
    # overrides applied on top of the chosen base config (for cluster runs:
    # seed sweeps via job arrays, custom checkpoint dirs, longer trainings)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--m", type=int, default=None,
                    help="sub-trajectory length the gradient flows through")
    ap.add_argument("--shock-ic-fraction", type=float, default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from alpha_model_last.eqx / opt_state.eqx "
                         "/ training_history.npz in --checkpoint-dir (e.g. "
                         "after a SLURM wall-time kill). Raise --epochs to "
                         "train further than the original run's target.")
    args = ap.parse_args()

    cfg = TrainConfig.light() if args.light else TrainConfig()
    overrides = {k: v for k, v in (
        ("epochs", args.epochs), ("seed", args.seed), ("lr", args.lr),
        ("m", args.m), ("shock_ic_fraction", args.shock_ic_fraction),
        ("checkpoint_dir", args.checkpoint_dir)) if v is not None}
    if overrides:
        cfg = replace(cfg, **overrides)
    print(f"config: {cfg}")
    train(cfg, resume=args.resume)
