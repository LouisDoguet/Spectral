"""Stage-1 supervised warm-start: imitate the Persson-Peraire indicator.

Runs BEFORE the differentiable MUSCL training. It:
  1. rolls the DGSEM solver driven by the Persson-Peraire (PP) alpha and
     collects the states along the way (so the network sees FORMED shocks, not
     just smooth initial conditions),
  2. regresses the network output toward alpha_PP on those states (broadcast
     across the P subcell interfaces for the nodal model).

The point is purely the cold start: PP is a competent, localized stabilizer, so
after this stage the very first Stage-2 rollout survives the forming shock and
produces a gradient instead of blowing up to NaN. It is a WARM START, not a
target -- Stage 2 then refines the policy toward the MUSCL-optimal alpha (and
should beat PP). Cheap: pure supervised regression, no MUSCL, no
backprop-through-time.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jax_dgsem.solver import rk4_step
from jax_dgsem.indicator import persson_peraire_alpha
from network.policy import alpha_features
from training.ic import draw_fourier_coeffs, sample_on_dgsem


def _pp_alpha(cfg, mesh):
    return lambda U: persson_peraire_alpha(U, mesh, alpha_max=cfg.alpha_max)


def collect_pp_states(cfg, mesh, rng, n_ics, samples_per_ic):
    """Run n_ics PP-driven rollouts and return their sampled states.

    Returns (M, 3, n_elem, Nn) of finite states along shock-forming rollouts."""
    pp_fn = _pp_alpha(cfg, mesh)
    stride = max(1, cfg.rollout_steps // samples_per_ic)
    n_out = cfg.rollout_steps // stride

    @eqx.filter_jit
    def rollout(U0):
        def inner(U, _):
            return rk4_step(U, pp_fn(U), cfg.dt, mesh), None

        def outer(U, _):
            U1, _ = jax.lax.scan(inner, U, None, length=stride)
            return U1, U1

        _, traj = jax.lax.scan(outer, U0, None, length=n_out)
        return traj

    states = []
    for _ in range(n_ics):
        coeffs = draw_fourier_coeffs(rng, cfg.n_fourier_modes, cfg.ic_amp,
                                     cfg.xL, cfg.xR,
                                     target_compression=cfg.target_compression,
                                     vel_modes=cfg.vel_modes)
        U0 = jnp.asarray(sample_on_dgsem(coeffs, mesh, cfg.xL))
        traj = np.concatenate([np.asarray(U0)[None], np.asarray(rollout(U0))])
        finite = np.isfinite(traj).all(axis=(1, 2, 3))
        states.append(traj[finite])
    return jnp.asarray(np.concatenate(states))


def pretrain_pp(model, cfg, mesh, seed, epochs, n_ics=None,
                samples_per_ic=32, batch_size=64):
    """Regress `model` toward the PP indicator; return the warm-started model."""
    n_ics = n_ics or cfg.K
    rng = np.random.default_rng(seed + 777)

    states = collect_pp_states(cfg, mesh, rng, n_ics, samples_per_ic)
    M = states.shape[0]

    # precompute inputs and PP targets once (they do not change)
    feats = jax.vmap(lambda U: alpha_features(U, mesh, cfg.model_type))(states)
    pp = jax.vmap(_pp_alpha(cfg, mesh))(states)              # (M, n_elem)
    out_shape = model(feats[0]).shape                        # (n_elem,) or (n_elem, P)
    if len(out_shape) == 2:                                  # nodal: broadcast to interfaces
        targets = jnp.broadcast_to(pp[:, :, None], (M,) + out_shape)
    else:
        targets = pp

    optimizer = optax.adam(cfg.pretrain_lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, feats_b, targets_b):
        def mse(m):
            return jnp.mean((jax.vmap(m)(feats_b) - targets_b) ** 2)
        loss, grads = eqx.filter_value_and_grad(mse)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    print(f"[pretrain] {M} PP states, imitating PP for {epochs} epochs")
    idx = np.arange(M)
    for e in range(epochs):
        rng.shuffle(idx)
        losses = []
        for i in range(0, M, batch_size):
            b = idx[i:i + batch_size]
            model, opt_state, loss = step(model, opt_state, feats[b], targets[b])
            losses.append(float(loss))
        if e % max(1, epochs // 10) == 0 or e == epochs - 1:
            print(f"[pretrain] epoch {e:3d}: MSE {np.mean(losses):.3e}")
    return model
