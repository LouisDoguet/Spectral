"""Watch the two solvers evolve from the SAME random-Fourier initial condition.

Generates one shared periodic Euler IC (draw_fourier_coeffs), applies it to
both grids, advances them in lockstep with the imposed dt, and animates the
fine MUSCL reference against the coarse hybrid-DGSEM solution (rho, u, p) plus
the per-element blending factor alpha.

Run from nn/ (or the repo root):
    python nn/main.py                       # DGSEM alpha from the trained NN if
                                            # a checkpoint exists, else PP
    python nn/main.py --alpha dg            # pure DG (no stabilization)
    python nn/main.py --alpha pp            # Persson-Peraire indicator
    python nn/main.py --alpha nn --model nn/training/checkpoints/alpha_model_best.eqx
    python nn/main.py --seed 7 --save out.gif   # different IC, save a GIF

--alpha nn falls back to PP with a warning if no checkpoint is found.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.solver import rk4_step
from jax_dgsem.physics import pressure
from jax_dgsem.indicator import (modal_energy, postprocess_alpha,
                                 persson_peraire_alpha)
from training.config import TrainConfig
from training.cost import uniform_projector
from training.ic import draw_fourier_coeffs, sample_on_dgsem, sample_on_muscl
from muscl.euler import (MusclEulerGrid, MusclEulerSolver,
                         pressure as muscl_pressure)

try:
    from vizstyle import apply_style, finish_axes, BLUE, YELLOW, INK, MUTED
except Exception:
    BLUE, YELLOW, INK, MUTED = "#2a78d6", "#eda100", "#111", "#888"
    def apply_style(): pass
    def finish_axes(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

PERIODIC = ("periodic", None)
PLOT_PTS = 12                    # DGSEM plotting points per element
N_FRAMES = 120                   # animation frames


# ---------------------------------------------------------------------------
# Alpha source for the DGSEM solve
# ---------------------------------------------------------------------------

def build_alpha_fn(kind, mesh, cfg, model_path):
    if kind == "dg":
        return lambda U: jnp.zeros(mesh.n_elem), "DGSEM (alpha = 0, pure DG)"
    if kind == "pp":
        return (lambda U: persson_peraire_alpha(U, mesh, alpha_max=cfg.alpha_max),
                "DGSEM + Persson-Peraire")
    if kind == "nn":
        from network.model import AlphaModel, load_model
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            print(f"[main] no checkpoint at {model_path}; falling back to PP")
            return build_alpha_fn("pp", mesh, cfg, model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        template = AlphaModel(meta["in_channels"], meta["width"],
                              meta["kernel_size"], meta["depth"],
                              key=jax.random.PRNGKey(0))
        model = load_model(model_path, template)
        amax = float(meta.get("alpha_max", cfg.alpha_max))

        def fn(U):
            return postprocess_alpha(model(modal_energy(U, mesh).T),
                                     alpha_max=amax, diffuse=cfg.alpha_diffuse,
                                     hard_clip=True)
        return fn, "DGSEM + NN policy"
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Run both solvers, storing frames
# ---------------------------------------------------------------------------

def run_both(cfg, seed, alpha_kind, model_path):
    basis = GLLBasis(cfg.P)
    mesh = Mesh1D(basis, cfg.n_elem, cfg.xL, cfg.xR, PERIODIC, PERIODIC)
    alpha_fn, dg_label = build_alpha_fn(alpha_kind, mesh, cfg, model_path)

    rng = np.random.default_rng(seed)
    coeffs = draw_fourier_coeffs(rng, cfg.n_fourier_modes, cfg.ic_amp,
                                 cfg.xL, cfg.xR,
                                 target_compression=cfg.target_compression,
                                 vel_modes=cfg.vel_modes)

    # --- DGSEM coarse solve: store state + alpha every `stride` steps -------
    stride = max(1, cfg.rollout_steps // N_FRAMES)
    n_out = cfg.rollout_steps // stride
    U0 = jnp.asarray(sample_on_dgsem(coeffs, mesh, cfg.xL))

    @eqx.filter_jit
    def dg_run(U0):
        def inner(U, _):
            return rk4_step(U, alpha_fn(U), cfg.dt, mesh), None

        def outer(U, _):
            U1, _ = jax.lax.scan(inner, U, None, length=stride)
            return U1, (U1, alpha_fn(U1))

        _, (traj, al) = jax.lax.scan(outer, U0, None, length=n_out)
        return traj, al

    dg_traj, dg_alpha = dg_run(U0)
    dg_traj = np.concatenate([np.asarray(U0)[None], np.asarray(dg_traj)])
    dg_alpha = np.concatenate([np.asarray(alpha_fn(U0))[None],
                               np.asarray(dg_alpha)])

    # --- MUSCL fine solve: same physical cadence ---------------------------
    grid = MusclEulerGrid(cfg.N_muscl, cfg.xL, cfg.xR)
    grid.set_state(sample_on_muscl(coeffs, grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps)
    muscl_traj = solver.trajectory(n_out, stride * cfg.muscl_substeps)

    # --- projections for plotting ------------------------------------------
    proj = uniform_projector(cfg.P, cfg.n_elem, PLOT_PTS)
    N_plot = cfg.n_elem * PLOT_PTS
    x_dg = cfg.xL + (np.arange(N_plot) + 0.5) * (cfg.xR - cfg.xL) / N_plot
    x_elem = cfg.xL + (np.arange(cfg.n_elem) + 0.5) * mesh.dx

    def dg_prims(U):
        Up = np.asarray(proj(jnp.asarray(U)))     # (3, N_plot)
        rho, rhou, E = Up
        u = rhou / rho
        p = (1.4 - 1.0) * (E - 0.5 * rhou * u)
        return rho, u, p

    times = np.arange(n_out + 1) * stride * cfg.dt
    return dict(mesh=mesh, x_dg=x_dg, x_elem=x_elem, x_muscl=grid.x,
                dg_traj=dg_traj, dg_alpha=dg_alpha, muscl_traj=muscl_traj,
                dg_prims=dg_prims, times=times, dg_label=dg_label, cfg=cfg)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate(data, save_path=None):
    apply_style()
    cfg = data["cfg"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    names = [("density", r"$\rho$", 0), ("velocity", "u", 1),
             ("pressure", "p", 2)]

    x_dg, x_m = data["x_dg"], data["x_muscl"]

    def muscl_prims(U):
        rho, rhou, E = U
        u = rhou / rho
        return rho, u, muscl_pressure(U)

    # y-limits from the whole MUSCL trajectory (robust reference range)
    allm = [muscl_prims(data["muscl_traj"][k]) for k in
            range(len(data["muscl_traj"]))]
    ylims = []
    for _, _, i in names:
        vals = np.concatenate([m[i] for m in allm])
        lo, hi = vals.min(), vals.max()
        pad = 0.1 * (hi - lo + 1e-9)
        ylims.append((lo - pad, hi + pad))

    lines = {}
    for (title, ylab, i), ax in zip(names, axes.ravel()):
        (lm,) = ax.plot([], [], color=INK, lw=1.3, label="MUSCL reference")
        (ld,) = ax.plot([], [], color=YELLOW, lw=2.0, label=data["dg_label"])
        lines[i] = (lm, ld)
        ax.set_xlim(cfg.xL, cfg.xR)
        ax.set_ylim(*ylims[i])
        ax.set_xlabel("x"); ax.set_ylabel(ylab); ax.set_title(title)
        if i == 0:
            ax.legend(loc="upper right")
        finish_axes(ax)

    ax_a = axes[1, 1]
    (la,) = ax_a.plot([], [], color=BLUE, lw=1.5)
    amax = max(0.05, float(np.nanmax(data["dg_alpha"])) * 1.15)
    ax_a.set_xlim(cfg.xL, cfg.xR); ax_a.set_ylim(0, amax)
    ax_a.set_xlabel("x"); ax_a.set_ylabel(r"$\alpha$")
    ax_a.set_title("blending factor")
    finish_axes(ax_a)

    def frame(k):
        muscl_vals = muscl_prims(data["muscl_traj"][k])
        finite = np.isfinite(data["dg_traj"][k]).all()
        dg_vals = data["dg_prims"](data["dg_traj"][k]) if finite else None
        for i in (0, 1, 2):
            lm, ld = lines[i]
            lm.set_data(x_m, muscl_vals[i])
            ld.set_data(x_dg, dg_vals[i]) if finite else ld.set_data([], [])
        ka = min(k, len(data["dg_alpha"]) - 1)
        la.set_data(data["x_elem"], data["dg_alpha"][ka])
        blow = "" if finite else "   [DGSEM blew up]"
        fig.suptitle(f"t = {data['times'][k]:.4f}   "
                     f"(DGSEM n_elem={cfg.n_elem} P={cfg.P}  vs  "
                     f"MUSCL N={cfg.N_muscl}){blow}",
                     color=INK, fontsize=13)
        return sum([list(l) for l in lines.values()], []) + [la]

    anim = FuncAnimation(fig, frame, frames=len(data["muscl_traj"]),
                         interval=60, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=20))
        print(f"[main] saved animation -> {save_path}")
    else:
        try:
            plt.show()
        except Exception:
            out = "nn/dgsem_vs_muscl.gif"
            anim.save(out, writer=PillowWriter(fps=20))
            print(f"[main] no display; saved animation -> {out}")
    return anim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", choices=["nn", "pp", "dg"], default="nn",
                    help="DGSEM blending source (default nn, falls back to pp)")
    ap.add_argument("--model",
                    default="nn/training/checkpoints/alpha_model_best.eqx")
    ap.add_argument("--seed", type=int, default=0, help="random IC seed")
    ap.add_argument("--save", default=None,
                    help="save the animation to this .gif instead of showing")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.rollout_steps is not None:
        from dataclasses import replace
        cfg = replace(cfg, rollout_steps=args.rollout_steps)

    data = run_both(cfg, args.seed, args.alpha, args.model)
    animate(data, args.save)


if __name__ == "__main__":
    main()
