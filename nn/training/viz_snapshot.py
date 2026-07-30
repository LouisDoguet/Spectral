"""During-training snapshot: DGSEM trainee (current model) vs MUSCL reference.

Saved every cfg.plot_every epochs so you can watch the DGSEM resolution track
(or fail to track) the fine MUSCL solution as the alpha policy learns. Shows
density at three times through the rollout, plus the network's alpha field.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_dgsem.solver import rk4_step
from jax_dgsem.indicator import postprocess_alpha
from training.cost import uniform_projector
from training.ic import sample_on_dgsem, sample_on_muscl
from muscl.euler import MusclEulerGrid, MusclEulerSolver

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from vizstyle import apply_style, finish_axes, BLUE, AQUA, YELLOW, INK, MUTED
except Exception:                         # vizstyle is optional
    BLUE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#1baf7a", "#eda100", "#111", "#888"
    def apply_style(): pass
    def finish_axes(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

from training.plotstyle import draw_elements

PLOT_PTS = 12


def _dgsem_rollout(model, mesh, U0, cfg):
    """Roll the DGSEM trainee from U0, deployment-mode alpha (hard clip), and
    return (trajectory (n+1,3,n_elem,Nn), alphas (n+1,n_elem))."""
    def alpha_of(U):
        from network.policy import apply_alpha
        return postprocess_alpha(apply_alpha(model, U, mesh, cfg.model_type),
                                 alpha_max=cfg.alpha_max, diffuse=cfg.alpha_diffuse,
                                 hard_clip=True)

    @eqx.filter_jit
    def run(U0):
        def body(U, _):
            a = alpha_of(U)
            return rk4_step(U, a, cfg.dt, mesh), (U, a)
        _, (traj, al) = jax.lax.scan(body, U0, None, length=cfg.rollout_steps)
        return traj, al

    traj, al = run(U0)
    traj = np.concatenate([np.asarray(U0)[None], np.asarray(traj)])
    al = np.concatenate([np.asarray(alpha_of(U0))[None], np.asarray(al)])
    return traj, al


def plot_snapshot(model, mesh, coeffs, cfg, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)
    apply_style()

    U0 = jnp.asarray(sample_on_dgsem(coeffs, mesh, cfg.xL))
    traj, alphas = _dgsem_rollout(model, mesh, U0, cfg)     # (n+1, ...)

    # MUSCL reference full field
    grid = MusclEulerGrid(cfg.N_muscl, cfg.xL, cfg.xR)
    grid.set_state(sample_on_muscl(coeffs, grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps)
    ref = solver.trajectory(cfg.rollout_steps, cfg.muscl_substeps)  # (n+1,3,N)
    x_muscl = grid.x

    # DGSEM plotting projection
    proj = uniform_projector(cfg.P, cfg.n_elem, PLOT_PTS)
    N_plot = cfg.n_elem * PLOT_PTS
    x_plot = cfg.xL + (np.arange(N_plot) + 0.5) * (cfg.xR - cfg.xL) / N_plot
    x_elem = cfg.xL + (np.arange(cfg.n_elem) + 0.5) * mesh.dx

    # alpha axis: nodal (T, n_elem, P) -> interface midpoints, flattened
    if alphas.ndim == 3:
        xn = np.asarray(mesh.node_positions(cfg.xL))
        x_alpha = (0.5 * (xn[:, :-1] + xn[:, 1:])).ravel()
        alphas = alphas.reshape(alphas.shape[0], -1)
    else:
        x_alpha = x_elem

    steps = [cfg.rollout_steps // 4, cfg.rollout_steps // 2, cfg.rollout_steps]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [2.4, 1]})
    finite = np.isfinite(traj).all(axis=(1, 2, 3))
    blow = None if finite.all() else int(np.argmin(finite))

    for j, s in enumerate(steps):
        ax = axes[0, j]
        t = s * cfg.dt
        ax.plot(x_muscl, ref[s, 0], color=INK, lw=1.3, label="MUSCL reference")
        if blow is None or s < blow:
            dg = np.asarray(proj(jnp.asarray(traj[s])))[0]
            ax.plot(x_plot, dg, color=YELLOW, lw=2.0, label="DGSEM + NN")
        else:
            ax.text(0.5, 0.5, f"DGSEM blew up\nat step {blow}",
                    transform=ax.transAxes, ha="center", color="#c0392b")
        draw_elements(ax, cfg.xL, cfg.xR, cfg.n_elem)
        ax.set_title(f"density,  t = {t:.4f}  (step {s})")
        ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
        if j == 0:
            ax.legend(loc="best")
        finish_axes(ax)

        # alpha field at this time
        axa = axes[1, j]
        s_a = min(s, len(alphas) - 1)
        axa.plot(x_alpha, alphas[s_a], color=BLUE, lw=1.5)
        axa.fill_between(x_alpha, 0, alphas[s_a], color=BLUE, alpha=0.15)
        draw_elements(axa, cfg.xL, cfg.xR, cfg.n_elem)
        axa.set_ylim(0, max(0.05, float(np.nanmax(alphas)) * 1.1))
        axa.set_xlabel("x"); axa.set_ylabel(r"$\alpha$")
        finish_axes(axa)

    fig.suptitle(f"epoch {epoch}: DGSEM (n_elem={cfg.n_elem}, P={cfg.P}) vs "
                 f"MUSCL (N={cfg.N_muscl})", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
