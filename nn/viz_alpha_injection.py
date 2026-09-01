"""Visualize how artificial viscosity (alpha) stabilizes DGSEM as a shock forms.

Runs three trajectories from the SAME initial condition (evaluation.compare.
setup_case, any of --case random/sod/lax/shu-osher): a fine MUSCL reference,
a raw DGSEM run (alpha = 0, unstable near the forming shock), and a DGSEM run
with dissipation injected (Persson-Peraire indicator or a trained NN policy).
Animates density (all three curves) and the dissipative scheme's alpha(x) so
you can see where/when viscosity is injected as the shock forms.

Run from the repo root, e.g.:
    .venv_spectral/bin/python nn/viz_alpha_injection.py --case shu-osher --alpha pp
    .venv_spectral/bin/python nn/viz_alpha_injection.py --case random --seed 7 --alpha nn \
        --model nn/training/checkpoints_AlphaNetO6_OPNO_C16/alpha_model_best.eqx

--seed only matters for --case random (it picks the Fourier draw); it is
ignored for the classic shock tubes. --save defaults to
nn/img/alpha_injection_<case>_<alpha>.mp4 when not given explicitly.

--alpha nn falls back to PP with a warning if no checkpoint is found.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from evaluation.compare import (CASES, setup_case, make_alpha_fn, run_scheme,
                                load_trained_model, _alpha_layout)
from network.policy import channels_from_meta
from training.config import TrainConfig
from training.cost import uniform_projector
from training.plotstyle import draw_elements
from muscl.euler import MusclEulerGrid, MusclEulerSolver
from vizstyle import apply_style, finish_axes, SCHEME_STYLE, INK

PLOT_PTS = 12       # DGSEM plotting points per element
N_FRAMES = 120      # animation frames


# ---------------------------------------------------------------------------
# Alpha source for the "add dissipation" DGSEM run
# ---------------------------------------------------------------------------

def resolve_dissipative_alpha(alpha_kind, model_path, mesh, cfg):
    """Returns (alpha_fn, scheme_key) for the dissipative DGSEM run, where
    scheme_key in {"pp", "nn"} selects color/label from vizstyle.SCHEME_STYLE."""
    if alpha_kind == "pp":
        return make_alpha_fn("pp", mesh, cfg, alpha_max=cfg.alpha_max), "pp"

    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        print(f"[viz_alpha_injection] no checkpoint at {model_path}; falling back to PP")
        return resolve_dissipative_alpha("pp", model_path, mesh, cfg)

    model, meta = load_trained_model(model_path)
    mtype = meta.get("model_type", "element")
    channels = channels_from_meta(meta)
    amax = float(meta.get("alpha_max", cfg.alpha_max))
    fn = make_alpha_fn("nn", mesh, cfg, model=model, mtype=mtype,
                       alpha_max=amax, channels=channels)
    return fn, "nn"


# ---------------------------------------------------------------------------
# Run all three solvers from the same IC
# ---------------------------------------------------------------------------

def run(cfg, case, seed, alpha_kind, model_path):
    cs = setup_case(case, cfg, seed)
    mesh, xL, xR, n_steps, tag = (cs["mesh"], cs["xL"], cs["xR"],
                                  cs["n_steps"], cs["tag"])
    U0 = cs["U0_dg"]

    stride = max(1, n_steps // N_FRAMES)
    n_out = n_steps // stride
    times = np.arange(n_out + 1) * stride * cfg.dt

    # --- MUSCL fine reference ------------------------------------------------
    grid = MusclEulerGrid(cfg.N_muscl, xL, xR)
    grid.set_state(cs["muscl_state"](grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps,
                              bc=cs["muscl_bc"])
    muscl_traj = solver.trajectory(n_out, stride * cfg.muscl_substeps)
    muscl_rho = muscl_traj[:, 0, :]

    # --- DGSEM: no stabilization (alpha = 0) ---------------------------------
    dg_fn = make_alpha_fn("dg", mesh, cfg)
    dg_traj, _, dg_blow = run_scheme(dg_fn, U0, mesh, cfg.dt, n_out, stride)

    # --- DGSEM: with dissipation ----------------------------------------------
    diss_fn, scheme_key = resolve_dissipative_alpha(alpha_kind, model_path, mesh, cfg)
    diss_traj, diss_alpha, diss_blow = run_scheme(diss_fn, U0, mesh, cfg.dt, n_out, stride)

    proj = uniform_projector(cfg.P, cfg.n_elem, PLOT_PTS)
    N_plot = cfg.n_elem * PLOT_PTS
    x_dg = xL + (np.arange(N_plot) + 0.5) * (xR - xL) / N_plot
    dg_rho = np.asarray(jax.vmap(proj)(jnp.asarray(dg_traj)))[:, 0]
    diss_rho = np.asarray(jax.vmap(proj)(jnp.asarray(diss_traj)))[:, 0]

    x_alpha, diss_alpha = _alpha_layout(diss_alpha, mesh, xL)

    return dict(cfg=cfg, tag=tag, mesh=mesh, xL=xL, xR=xR, times=times,
               x_muscl=grid.x, muscl_rho=muscl_rho,
               x_dg=x_dg, dg_rho=dg_rho, dg_blow=dg_blow,
               diss_rho=diss_rho, diss_blow=diss_blow, scheme_key=scheme_key,
               x_alpha=x_alpha, diss_alpha=diss_alpha)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate(data, save_path):
    apply_style()
    cfg, xL, xR, times = data["cfg"], data["xL"], data["xR"], data["times"]
    scheme_key = data["scheme_key"]
    dg_style, diss_style = SCHEME_STYLE["dg"], SCHEME_STYLE[scheme_key]

    dg_finite = np.isfinite(data["dg_rho"]).all(axis=1)
    diss_finite = np.isfinite(data["diss_rho"]).all(axis=1)
    muscl_finite = np.isfinite(data["muscl_rho"]).all(axis=1)

    fig, (ax_rho, ax_a) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1]))

    finite_chunks = [a[m] for a, m in
                     ((data["muscl_rho"], muscl_finite),
                      (data["dg_rho"], dg_finite),
                      (data["diss_rho"], diss_finite)) if m.any()]
    finite_rho = np.concatenate([c.ravel() for c in finite_chunks]) \
        if finite_chunks else np.array([0.0, 1.0])
    lo, hi = finite_rho.min(), finite_rho.max()
    pad = 0.1 * (hi - lo + 1e-9)

    (line_muscl,) = ax_rho.plot([], [], color=INK, lw=1.4, label="MUSCL reference")
    (line_dg,) = ax_rho.plot([], [], color=dg_style["color"], ls=dg_style["ls"],
                             lw=2.0, label=dg_style["label"])
    (line_diss,) = ax_rho.plot([], [], color=diss_style["color"], ls=diss_style["ls"],
                               lw=2.0, label=diss_style["label"])
    ax_rho.set_xlim(xL, xR); ax_rho.set_ylim(lo - pad, hi + pad)
    ax_rho.set_ylabel(r"density $\rho$")
    ax_rho.legend(loc="upper right")
    draw_elements(ax_rho, xL, xR, cfg.n_elem)
    finish_axes(ax_rho)

    (line_alpha,) = ax_a.plot([], [], color=diss_style["color"], lw=1.6)
    amax = max(0.05, float(np.nanmax(data["diss_alpha"])) * 1.15)
    ax_a.set_xlim(xL, xR); ax_a.set_ylim(0, amax)
    ax_a.set_xlabel("x"); ax_a.set_ylabel(r"$\alpha$")
    ax_a.set_title(diss_style["label"])
    draw_elements(ax_a, xL, xR, cfg.n_elem)
    finish_axes(ax_a)

    def frame(k):
        artists = []
        if muscl_finite[k]:
            line_muscl.set_data(data["x_muscl"], data["muscl_rho"][k])
            artists.append(line_muscl)

        blow_bits = []
        if dg_finite[k]:
            line_dg.set_data(data["x_dg"], data["dg_rho"][k])
            artists.append(line_dg)
        elif data["dg_blow"] is not None:
            blow_bits.append(f"{dg_style['label']} blew up at t={times[data['dg_blow']]:.4f}")

        if diss_finite[k]:
            line_diss.set_data(data["x_dg"], data["diss_rho"][k])
            line_alpha.set_data(data["x_alpha"], data["diss_alpha"][k])
            artists += [line_diss, line_alpha]
        elif data["diss_blow"] is not None:
            blow_bits.append(f"{diss_style['label']} blew up at t={times[data['diss_blow']]:.4f}")

        suffix = "   [" + "; ".join(blow_bits) + "]" if blow_bits else ""
        fig.suptitle(f"{data['tag']}   t = {times[k]:.4f}   (DGSEM P={cfg.P}, "
                    f"n_elem={cfg.n_elem} vs MUSCL N={cfg.N_muscl}){suffix}",
                    color=INK, fontsize=15)
        return artists

    anim = FuncAnimation(fig, frame, frames=len(times), interval=60, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer=FFMpegWriter(fps=20))
    plt.close(fig)
    return anim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="random", choices=CASES,
                    help="which initial condition to animate")
    ap.add_argument("--alpha", choices=["pp", "nn"], default="pp",
                    help="dissipation source for the stabilized DGSEM curve")
    ap.add_argument("--model",
                    default="nn/training/benchmark/opno_search/full/W128_D3_OH128_OC4_FH48_P6/alpha_model_best.eqx",
                    help="checkpoint used when --alpha nn")
    ap.add_argument("--seed", type=int, default=0,
                    help="IC seed (only used by --case random)")
    ap.add_argument("--save", default=None,
                    help="output path (default: nn/img/alpha_injection_<case>_<alpha>.mp4)")
    args = ap.parse_args()

    cfg = TrainConfig()
    data = run(cfg, args.case, args.seed, args.alpha, args.model)
    save_path = args.save or f"nn/img/alpha_injection_{data['tag']}_{args.alpha}.mp4"
    animate(data, save_path)
    print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()
