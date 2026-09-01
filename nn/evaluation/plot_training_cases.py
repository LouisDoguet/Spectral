"""Static, publication-style overview of the training/eval cases the alpha
policy sees: how each case's initial state develops a discontinuity over its
physical horizon, on the fine-grid MUSCL reference ONLY (no DGSEM / PP / NN
scheme -- this is a "what is the training data" figure, not a stabilization
comparison; see evaluation/compare.py for that).

For now this shows --n-random distinct draws of the periodic random-Fourier
training IC (training.ic.draw_fourier_coeffs), run for the same physical
horizon (rollout_steps * dt) training.config.TrainConfig defaults use; the
classic validation cases (sod/lax/shu-osher, reusing muscl/reference_gif.py's
exact ICs via run_classic) are meant to be added as extra rows later --
extend default_rows()'s returned list with ("classic", "sod", "Sod") etc.

    .venv_spectral/bin/python nn/evaluation/plot_training_cases.py
    -> nn/img/training_cases.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from muscl.euler import (MusclEulerGrid, MusclEulerSolver, GAMMA,
                         max_wave_speed)
from muscl.reference_gif import CASES as CLASSIC_CASES, build_ic
from training.config import TrainConfig
from training.ic import draw_fourier_coeffs, sample_on_muscl
from vizstyle import apply_style, finish_axes, INK, INK_2, MUTED

# Sequential "time" ramp -- deliberately a different hue from vizstyle's blue
# alpha ramp: this figure has no alpha/scheme content, so reusing that ramp
# would wrongly imply a shared meaning. Light -> dark = early -> late time, so
# the eye reads the discontinuity forming as the curves darken.
_TIME_RAMP = ["#fde6c2", "#fbd096", "#f7b76c", "#f19c49", "#e37f2d",
             "#c8621d", "#a84815", "#7e330f"]
TIME_CMAP = LinearSegmentedColormap.from_list("seq_time", _TIME_RAMP)

N_SNAPSHOTS = 6     # curves per subplot, including t=0
N_CELLS = 1000       # fine MUSCL grid for this illustration
SEED = 12345        # matches evaluation/benchmark.py's EVAL_SEED
CFL = 0.4
SAFETY = 1.6        # margin for the post-shock wave-speed increase
VAR_KEYS = ("rho", "u", "p")
VAR_LABEL = {"rho": "density ρ", "u": "velocity u", "p": "pressure p"}


def default_rows(seed, n_random=2):
    """n_random distinct random-Fourier draws (seed, seed+1, ...). Each row is
    (kind, arg, label): kind="random" draws with arg=seed; kind="classic" reads
    CLASSIC_CASES[arg]. Extend this list with ("classic", "sod", "Sod") etc.
    once the validation cases join the figure."""
    return [("random", np.random.random_integers(low=0, high=10e10), "Random Fourier"),
            ("random", np.random.random_integers(low=0, high=10e10), "Random Fourier")]
    return [("classic", "sod", f"SOD"),
            ("classic", "lax", f"Lax"),
            ("classic", "shu-osher", f"Shu-Osher")]


# ---------------------------------------------------------------------------
# Reference MUSCL runs (adaptive-CFL dt, landed exactly on the horizon T)
# ---------------------------------------------------------------------------

def _run_to_horizon(grid, T, bc, frames):
    """Step `grid`'s current state to time T, snapshotting `frames` times
    (+ the IC), via a CFL-stable dt sized from the initial wave speed."""
    a0 = SAFETY * float(np.max(max_wave_speed(grid.U)))
    total = max(1, int(np.ceil(T / (CFL * grid.dx / a0))))
    inner = max(1, int(np.ceil(total / frames)))
    n_outer = int(np.ceil(total / inner))
    dt = T / (n_outer * inner)
    solver = MusclEulerSolver(grid, dt=dt, bc=bc)
    traj = solver.trajectory(n_outer, inner)          # (n_outer+1, 3, N)
    times = np.linspace(0.0, T, n_outer + 1)
    return times, traj


def run_classic(name, frames):
    """(x, times, traj) for a classic case (sod/lax/shu-osher), the exact IC
    used by reference_gif.py's MUSCL reference animations."""
    c = CLASSIC_CASES[name]
    grid = MusclEulerGrid(N_CELLS, c["xL"], c["xR"])
    grid.set_state(build_ic(name, grid.x))
    times, traj = _run_to_horizon(grid, c["T"], c["bc"], frames)
    return grid.x, times, traj


def run_random(frames, seed=SEED):
    """(x, times, traj) for the periodic random-Fourier training IC, run for
    the same physical horizon (rollout_steps * dt) the closed-loop training
    targets (TrainConfig defaults)."""
    cfg = TrainConfig()
    xL, xR = cfg.xL, cfg.xR
    coeffs = draw_fourier_coeffs(np.random.default_rng(seed), cfg.n_fourier_modes,
                                 cfg.ic_amp, xL, xR,
                                 target_compression=cfg.target_compression,
                                 vel_modes=cfg.vel_modes)
    grid = MusclEulerGrid(N_CELLS, xL, xR)
    grid.set_state(sample_on_muscl(coeffs, grid))
    T = cfg.rollout_steps * cfg.dt
    times, traj = _run_to_horizon(grid, T, "periodic", frames)
    return grid.x, times, traj


def primitives(traj):
    """traj (T, 3, N) -> dict(rho, u, p), each (T, N)."""
    rho, rhou, E = traj[:, 0], traj[:, 1], traj[:, 2]
    u = rhou / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rhou * u)
    return {"rho": rho, "u": u, "p": p}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(n_snapshots=N_SNAPSHOTS, seed=SEED, rows=None):
    apply_style()
    rows = default_rows(seed) if rows is None else rows

    data = []
    for kind, arg, label in rows:
        x, times, traj = (run_random(n_snapshots - 1, arg) if kind == "random"
                          else run_classic(arg, n_snapshots - 1))
        data.append(dict(x=x, times=times, label=label, **primitives(traj)))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12.5, 2.6 * n_rows + 1.4))
    if n_rows == 1:
        axes = axes[None, :]
    fig.subplots_adjust(left=0.095, right=0.90, top=0.88, bottom=0.075,
                        hspace=0.55, wspace=0.28)

    for i, d in enumerate(data):
        t_frac = d["times"] / d["times"][-1]
        lws = np.linspace(1.1, 2.4, len(t_frac))
        for j, key in enumerate(VAR_KEYS):
            ax = axes[i, j]
            for k in range(len(t_frac)):
                ax.plot(d["x"], d[key][k], color=TIME_CMAP(t_frac[k]),
                        lw=lws[k], solid_capstyle="round", zorder=2 + k)
            finish_axes(ax)
            ax.set_xlim(d["x"][0], d["x"][-1])
            ax.margins(y=0.08)
            if i == 0:
                ax.set_title(VAR_LABEL[key], fontweight="bold",
                            color=INK, pad=10)
            if i == n_rows - 1:
                ax.set_xlabel("x")

    # Row labels, outside the axes on the left.
    for i, d in enumerate(data):
        bbox = axes[i, 0].get_position()
        y = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.015, y, d["label"], rotation=90, ha="center", va="center",
                 fontsize=16, fontweight="bold", color=INK)

    # Shared "time" colorbar (the only legend this figure needs: one series,
    # one sequential encoding).
    cax = fig.add_axes((0.925, 0.14, 0.014, 0.68))
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=TIME_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("t / T", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=14, color=MUTED, labelcolor=MUTED)

    #fig.suptitle("Reference MUSCL solutions used to train the α-policy",
    #            fontsize=15.5, fontweight="bold", color=INK, x=0.47, y=0.975)
    #fig.text(0.47, 0.935,
    #         "Density, velocity, and pressure as each case's discontinuity "
    #         "develops — fine-grid MUSCL reference only",
    #         ha="center", fontsize=10, color=INK_2)

    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="nn/img/training_cases.png")
    ap.add_argument("--snapshots", type=int, default=N_SNAPSHOTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=2,
                    help="number of random-Fourier draws (rows seed, seed+1, ...)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = default_rows(args.seed, args.n_random)
    fig = make_figure(args.snapshots, args.seed, rows=rows)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_out)
    print(f"saved {pdf_out}")


if __name__ == "__main__":
    main()
