"""Visualize what the input preconditioner does to the network's feature space.

Builds the REAL nodal data channels (network.policy.nodal_data) on a shock state,
then applies the REAL whitening the model uses (network.model.whiten_rows), and
shows the feature space before vs after preconditioning:

  col 1  feature channels over x           (with element boundaries)
  col 2  channel-i vs channel-j scatter    (points = integration nodes)
  col 3  feature covariance matrix         (raw -> whitened is ~ identity)

The one-hot position block is NOT preconditioned -- it is appended AFTER the
data channels are whitened -- so only the data channels are shown here.

Publication-quality via plotstyle. Standalone:

    python nn/training/viz_precondition.py --case sod --out nn/img/precondition.pdf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import plotstyle
from plotstyle import PALETTE, draw_elements

from jax_dgsem import GLLBasis, Mesh1D
from network.policy import nodal_channel_names, nodal_data
from network.model import whiten_rows
from training.config import TrainConfig
from training.ic import (primitives_to_conservative, draw_fourier_coeffs,
                         sample_on_dgsem)

# "random" = a smooth Fourier IC (the training distribution: dense, richly varied
# features, so the whitening story is vivid). The classic shock ICs (primitive,
# domains match lib/{SOD,lax,shu_osher}.cpp) are also available.
CASES = {
    "random": dict(xL=0.0, xR=1.0),
    "sod": dict(xL=0.0, xR=1.0, x0=0.5, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1)),
    "lax": dict(xL=0.0, xR=1.0, x0=0.5, left=(0.445, 0.698, 3.528), right=(0.5, 0.0, 0.571)),
    "shu-osher": dict(xL=-5.0, xR=5.0, x0=-4.0, left=(3.857143, 2.629369, 10.33333)),
}


def _pp_rollout(U, mesh, steps):
    """Advance `steps` RK4 steps with Persson-Peraire-stabilized alpha, so a
    steep/near-shock profile forms and BOTH data channels become informative
    (the energy channel is ~flat on a smooth t=0 IC)."""
    from jax_dgsem.indicator import persson_peraire_alpha
    from jax_dgsem.solver import rk4_step
    dt = TrainConfig().dt

    @jax.jit
    def roll(U):
        def body(U, _):
            a = persson_peraire_alpha(U, mesh, alpha_max=1.0)
            return rk4_step(U, a, dt, mesh), None
        Uf, _ = jax.lax.scan(body, U, None, length=steps)
        return Uf
    return roll(U)


def build_state(case, mesh, xL, seed=0, steps=0):
    """Conservative state on the mesh GLL nodes -> (3, n_elem, Nn). When steps>0
    the initial state is rolled forward (PP-stabilized) to a shocked profile."""
    c = CASES[case]
    if case == "random":
        cfg = TrainConfig()
        coeffs = draw_fourier_coeffs(np.random.default_rng(seed),
                                     cfg.n_fourier_modes, cfg.ic_amp, xL, c["xR"],
                                     target_compression=cfg.target_compression,
                                     vel_modes=cfg.vel_modes)
        U = jnp.asarray(sample_on_dgsem(coeffs, mesh, xL))
    else:
        x = np.asarray(mesh.node_positions(xL))
        if case == "shu-osher":
            rhoL, uL, pL = c["left"]
            rho = np.where(x < c["x0"], rhoL, 1.0 + 0.2 * np.sin(5.0 * x))
            u = np.where(x < c["x0"], uL, 0.0)
            p = np.where(x < c["x0"], pL, 1.0)
        else:
            (rhoL, uL, pL), (rhoR, uR, pR) = c["left"], c["right"]
            left = x < c["x0"]
            rho = np.where(left, rhoL, rhoR)
            u = np.where(left, uL, uR)
            p = np.where(left, pL, pR)
        U = jnp.asarray(primitives_to_conservative(rho, u, p))
    return _pp_rollout(U, mesh, steps) if steps > 0 else U


def _cov(X):
    """Feature covariance (F, F) of X = (N_points, F), matching whiten_rows."""
    Xc = X - X.mean(0, keepdims=True)
    return (Xc.T @ Xc) / X.shape[0]


def _panel_lines(ax, x, X, names, xL, xR, n_elem, title):
    draw_elements(ax, xL, xR, n_elem)
    order = np.argsort(x)
    for j, nm in enumerate(names):
        ax.plot(x[order], X[order, j], color=PALETTE[j % len(PALETTE)],
                lw=1.2, label=nm)
    ax.set_xlabel("x"); ax.set_ylabel("feature value")
    ax.set_title(title); ax.legend(loc="best", ncol=1)


def _panel_scatter(ax, X, x, names, title, equal_unit=False):
    sc = ax.scatter(X[:, 0], X[:, 1], c=x, cmap="cividis", s=10,
                    edgecolor="none", alpha=0.85)
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_title(title)
    ax.axhline(0, color="#bbbbbb", lw=0.5, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.5, zorder=0)
    if equal_unit:
        ax.set_aspect("equal", "box")
        th = np.linspace(0, 2 * np.pi, 100)      # unit circle = unit-variance ref
        ax.plot(np.cos(th), np.sin(th), color="#c0392b", lw=0.8, ls=(0, (4, 2)))
    return sc


def _panel_cov(ax, C, names, title):
    v = float(np.max(np.abs(C)))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    ax.set_title(title); ax.grid(False)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2g}", ha="center", va="center",
                    color="#000000", fontsize=11)
    return im


def make_figure(case, P, n_elem, out_path, steps=0, seed=0):
    plotstyle.apply()
    basis = GLLBasis(P)
    c = CASES[case]
    mesh = Mesh1D(basis, n_elem, c["xL"], c["xR"])
    U = build_state(case, mesh, c["xL"], seed=seed, steps=steps)

    names = nodal_channel_names(P)                  # len == C (spectrum expands)
    data = np.asarray(nodal_data(U, mesh))          # (C, n_elem, Nn)
    C = data.shape[0]
    rows = data.reshape(C, -1)                       # (C, N_points)  = input injected
    white = np.asarray(whiten_rows(jnp.asarray(rows)))
    X, Xw = rows.T, white.T                          # (N_points, C)

    xnodes = np.asarray(mesh.node_positions(c["xL"])).ravel()   # (N_points,)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), constrained_layout=True)

    # row 0: raw injected features -----------------------------------------
    _panel_lines(axes[0, 0], xnodes, X, names, c["xL"], c["xR"], n_elem, "(a)")
    scatter_names = names[:2] if C >= 2 else names * 2
    if C >= 2:
        _panel_scatter(axes[0, 1], X, xnodes, scatter_names, "(b)")
    else:
        axes[0, 1].axis("off")
    _panel_cov(axes[0, 2], _cov(X), names, "(c)")

    # row 1: after the preconditioner --------------------------------------
    _panel_lines(axes[1, 0], xnodes, Xw, names, c["xL"], c["xR"], n_elem, "(d)")
    if C >= 2:
        sc = _panel_scatter(axes[1, 1], Xw, xnodes, scatter_names,
                            "(e)", equal_unit=True)
        cb = fig.colorbar(sc, ax=axes[0, 1], location="right", pad=0.02,
                          fraction=0.046)
        cb.set_label("node position x")
    else:
        axes[1, 1].axis("off")
    _panel_cov(axes[1, 2], _cov(Xw), names, "(f)")

    for ax in (axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]):
        plotstyle.despine(ax)
    plotstyle.save(out_path, fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="random", choices=list(CASES))
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--n_elem", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None,
                    help="PP-stabilized rollout steps before sampling features "
                         "(default: 400 for 'random' so a shock forms, else 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output path (.pdf/.png/.svg)")
    args = ap.parse_args()
    steps = args.steps if args.steps is not None else (400 if args.case == "random" else 0)
    out = args.out or f"nn/img/precondition_{args.case}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make_figure(args.case, args.P, args.n_elem, out, steps=steps, seed=args.seed)


if __name__ == "__main__":
    main()
