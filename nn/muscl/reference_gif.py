"""Animate the fine-grid MUSCL *reference* solution for the classic 1D Euler
benchmarks (Sod, Lax, Shu-Osher) and save each as a GIF.

Uses the same reference solver the alpha-policy training targets
(muscl.euler.MusclEulerSolver: minmod + Rusanov + SSP-RK2), with the exact
initial conditions of the C++ benchmark cases (lib/{SOD,lax,shu_osher}.cpp) and
transmissive (zero-gradient) boundaries.

    python nn/muscl/reference_gif.py --case all --outdir nn/img
    python nn/muscl/reference_gif.py --case sod --N 4096 --frames 200

A stable imposed dt is picked from the initial-field CFL (the solver does not
adapt), with margin for the post-shock speed-up.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from muscl.euler import (MusclEulerGrid, MusclEulerSolver, GAMMA,
                         pressure as muscl_pressure, max_wave_speed)

try:
    from vizstyle import apply_style, BLUE, YELLOW, INK
except Exception:
    BLUE, YELLOW, INK = "#2a78d6", "#eda100", "#111"
    def apply_style(): pass


def prim_to_cons(rho, u, p):
    """(rho, u, p) -> conservative U (3, N)."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E])


# Canonical benchmark cases (values/domains match lib/{SOD,lax,shu_osher}.cpp).
CASES = {
    "sod": dict(
        xL=0.0, xR=1.0, T=0.2, bc="transmissive",
        left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), x0=0.5),
    "lax": dict(
        xL=0.0, xR=1.0, T=0.16, bc="transmissive",
        left=(0.445, 0.698, 3.528), right=(0.5, 0.0, 0.571), x0=0.5),
    "shu-osher": dict(
        xL=-5.0, xR=5.0, T=1.8, bc="transmissive",
        left=(3.857143, 2.629369, 10.33333), x_shock=-4.0),
}


def build_ic(name, x):
    c = CASES[name]
    if name == "shu-osher":
        rhoL, uL, pL = c["left"]
        rho = np.where(x < c["x_shock"], rhoL, 1.0 + 0.2 * np.sin(5.0 * x))
        u = np.where(x < c["x_shock"], uL, 0.0)
        p = np.where(x < c["x_shock"], pL, 1.0)
    else:
        rhoL, uL, pL = c["left"]
        rhoR, uR, pR = c["right"]
        left = x < c["x0"]
        rho = np.where(left, rhoL, rhoR)
        u = np.where(left, uL, uR)
        p = np.where(left, pL, pR)
    return prim_to_cons(rho, u, p)


def run_case(name, N, frames, cfl=0.4, safety=1.6):
    """Return (x, times, trajectory) with trajectory shape (frames+1, 3, N)."""
    c = CASES[name]
    grid = MusclEulerGrid(N, c["xL"], c["xR"])
    U0 = build_ic(name, grid.x)
    grid.set_state(U0)

    # Imposed-dt stability: base dt on the initial fastest speed with margin for
    # the post-shock acceleration, then land exactly on T.
    a0 = safety * float(np.max(max_wave_speed(U0)))
    total = int(np.ceil(c["T"] / (cfl * grid.dx / a0)))
    inner = max(1, int(np.ceil(total / frames)))
    n_outer = int(np.ceil(total / inner))
    dt = c["T"] / (n_outer * inner)

    solver = MusclEulerSolver(grid, dt=dt, bc=c["bc"])
    traj = solver.trajectory(n_outer, inner)          # (n_outer+1, 3, N)
    times = np.linspace(0.0, c["T"], n_outer + 1)
    return grid.x, times, traj


def animate(name, x, times, traj, out_path, fps=20):
    prims = [(traj[k, 0],
              traj[k, 1] / traj[k, 0],
              muscl_pressure(traj[k])) for k in range(len(traj))]
    labels = ["Density  ρ", "Velocity  u", "Pressure  p"]
    colors = [BLUE, YELLOW, INK]

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    lines = []
    for i, (ax, lab, col) in enumerate(zip(axes, labels, colors)):
        (ln,) = ax.plot([], [], color=col, lw=1.6)
        lines.append(ln)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)
        allv = np.concatenate([p[i] for p in prims])   # this channel, all frames
        pad = 0.05 * (allv.max() - allv.min() + 1e-9)
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.set_xlim(x[0], x[-1])
    axes[-1].set_xlabel("x")
    title = fig.suptitle("")

    def update(k):
        for i, ln in enumerate(lines):
            ln.set_data(x, prims[k][i])
        title.set_text(f"MUSCL reference — {name}   (N={len(x)},  t = {times[k]:.3f})")
        return (*lines, title)

    anim = FuncAnimation(fig, update, frames=len(traj), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {out_path}  ({len(traj)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="all",
                    choices=["all", "sod", "lax", "shu-osher"])
    ap.add_argument("--N", type=int, default=2048, help="MUSCL cells (fine grid)")
    ap.add_argument("--frames", type=int, default=160, help="animation frames")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--outdir", default="nn/img", help="output directory for GIFs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = ["sod", "lax", "shu-osher"] if args.case == "all" else [args.case]
    for name in names:
        print(f"[{name}] running MUSCL reference (N={args.N}) ...")
        x, times, traj = run_case(name, args.N, args.frames)
        out = os.path.join(args.outdir, f"{name}_muscl_reference.gif")
        animate(name, x, times, traj, out, fps=args.fps)


if __name__ == "__main__":
    main()
