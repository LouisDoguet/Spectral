"""Three-way stabilization comparison against a fine MUSCL Euler reference:
pure DGSEM vs hybrid+Persson-Peraire vs hybrid+trained-NN, from the SAME
initial condition.

  - "dg" : alpha = 0 everywhere -> the raw high-order scheme, which oscillates
           at the shock and typically blows up (the stability problem),
  - "pp" : the built-in Persson-Peraire indicator (the hand-tuned fix),
  - "nn" : the trained alpha policy in deployment mode (hard clip + cap). Works
           for both model types (per-element or nodal per-interface).

Cases (--case):
  random        periodic shared random-Fourier shock-forming IC (--seed picks it)
  sod / lax     classic Riemann shock tubes on [0,1], wall/transmissive BCs
  shu-osher     shock / entropy-wave interaction on [-5,5]

The reference is always the fine MUSCL Euler solve; both grids use the imposed
dt (cfg.dt, no CFL) and the same IC.

Outputs (into --outdir):
  <case>_comparison.png       density + shock zoom + error growth + alpha budget
  <case>_alpha_spacetime.png  where/when PP and NN inject dissipation
  <case>_metrics.csv          final errors, mean alpha, blow-up time

Run from the repo root, e.g.:
    .venv_spectral/bin/python nn/evaluation/compare.py --case sod \
        --model nn/training/checkpoints/alpha_model_best.eqx
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.solver import rk4_step
from jax_dgsem.indicator import persson_peraire_alpha, postprocess_alpha
from network.model import load_model
from network.policy import apply_alpha, build_from_meta, channels_from_meta
from training.config import TrainConfig
from training.cost import uniform_projector
from training.ic import (draw_fourier_coeffs, sample_on_dgsem, sample_on_muscl,
                         primitives_to_conservative)
from muscl.euler import MusclEulerGrid, MusclEulerSolver
from vizstyle import (apply_style, finish_axes, SCHEME_STYLE, ALPHA_CMAP,
                      INK, INK_2)
from training.plotstyle import draw_elements

SCHEMES = ("dg", "pp", "nn")
PERIODIC = ("periodic", None)
PLOT_PTS = 12                 # DGSEM plotting points per element
CASES = ("random", "sod", "lax", "shu-osher")

# classic shock tubes: primitive (rho, u, p) left/right states, split point,
# domain and standard final time
CLASSIC = {
    "sod": dict(domain=(0.0, 1.0), left=(1.0, 0.0, 1.0),
                right=(0.125, 0.0, 0.1), x0=0.5, T=0.2),
    "lax": dict(domain=(0.0, 1.0), left=(0.445, 0.698, 3.528),
                right=(0.5, 0.0, 0.571), x0=0.5, T=0.16),
    "shu-osher": dict(domain=(-5.0, 5.0), T=1.8),
}


# ---------------------------------------------------------------------------
# Case setup (initial condition on BOTH grids + BCs + horizon)
# ---------------------------------------------------------------------------

def classic_primitives(case, x, delta):
    """(rho, u, p) of a classic case at arbitrary x, tanh-smoothed by delta."""
    if case == "shu-osher":
        rhoL, uL, pL = 3.857143, 2.629369, 10.33333       # post-shock state
        rhoR = 1.0 + 0.2 * np.sin(5.0 * x)                # entropy wave
        s = 0.5 * (1.0 - np.tanh((x + 4.0) / delta))      # x0 = -4
        return rhoR + (rhoL - rhoR) * s, uL * s, 1.0 + (pL - 1.0) * s
    spec = CLASSIC[case]
    L, R, x0 = spec["left"], spec["right"], spec["x0"]
    s = 0.5 * (1.0 - np.tanh((x - x0) / delta))
    return (R[0] + (L[0] - R[0]) * s, R[1] + (L[1] - R[1]) * s,
            R[2] + (L[2] - R[2]) * s)


def setup_case(case, cfg, seed):
    """Returns a dict: mesh (DGSEM, with BCs), U0_dg, muscl_state(grid) builder,
    muscl_bc, xL, xR, n_steps, tag."""
    basis = GLLBasis(cfg.P)
    if case == "random":
        xL, xR = cfg.xL, cfg.xR
        mesh = Mesh1D(basis, cfg.n_elem, xL, xR, PERIODIC, PERIODIC)
        coeffs = draw_fourier_coeffs(
            np.random.default_rng(seed), cfg.n_fourier_modes, cfg.ic_amp,
            xL, xR, target_compression=cfg.target_compression,
            vel_modes=cfg.vel_modes)
        return dict(mesh=mesh, U0_dg=jnp.asarray(sample_on_dgsem(coeffs, mesh, xL)),
                    muscl_state=lambda grid: sample_on_muscl(coeffs, grid),
                    muscl_bc="periodic", xL=xL, xR=xR,
                    n_steps=cfg.rollout_steps, tag=f"random_seed{seed}")

    # classic shock tube: wall BCs on DGSEM (fixed Dirichlet = IC endpoints),
    # transmissive on the MUSCL reference
    xL, xR = CLASSIC[case]["domain"]
    delta = 2.0 * (xR - xL) / cfg.n_elem                  # ~2 coarse cells
    zero = np.zeros(3)
    mesh0 = Mesh1D(basis, cfg.n_elem, xL, xR, ("wall", zero), ("wall", zero))
    xn = np.asarray(mesh0.node_positions(xL))
    U0_dg = np.asarray(primitives_to_conservative(
        *classic_primitives(case, xn, delta)))
    mesh = eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state), mesh0,
                       (jnp.asarray(U0_dg[:, 0, 0]), jnp.asarray(U0_dg[:, -1, -1])))
    n_steps = int(round(CLASSIC[case]["T"] / cfg.dt))
    return dict(mesh=mesh, U0_dg=jnp.asarray(U0_dg),
                muscl_state=lambda grid: primitives_to_conservative(
                    *classic_primitives(case, grid.x, delta)),
                muscl_bc="transmissive", xL=xL, xR=xR,
                n_steps=n_steps, tag=case)


# ---------------------------------------------------------------------------
# Alpha sources and scheme runs
# ---------------------------------------------------------------------------

def make_alpha_fn(scheme, mesh, cfg, model=None, mtype="element", alpha_max=1.0,
                  channels=None):
    if scheme == "dg":
        return lambda U: jnp.zeros(mesh.n_elem)
    if scheme == "pp":
        return lambda U: persson_peraire_alpha(U, mesh, alpha_max=alpha_max)
    if scheme == "nn":
        def fn(U):
            return postprocess_alpha(
                apply_alpha(model, U, mesh, mtype, channels),
                alpha_min=0.001, alpha_max=alpha_max,
                diffuse=cfg.alpha_diffuse, hard_clip=True)
        return fn
    raise ValueError(scheme)


def run_scheme(alpha_fn, U0, mesh, dt, n_out, stride):
    """Advance n_out*stride RK4 steps, snapshotting every stride. Returns
    (traj, alphas, blowup index or None). A blown-up run keeps its NaN tail."""

    @eqx.filter_jit
    def _run(U0):
        def inner(U, _):
            return rk4_step(U, alpha_fn(U), dt, mesh), None

        def outer(U, _):
            U1, _ = jax.lax.scan(inner, U, None, length=stride)
            return U1, (U1, alpha_fn(U1))

        _, (traj, al) = jax.lax.scan(outer, U0, None, length=n_out)
        return traj, al

    traj, al = _run(U0)
    traj = np.concatenate([np.asarray(U0)[None], np.asarray(traj)])
    al = np.concatenate([np.asarray(alpha_fn(U0))[None], np.asarray(al)])
    finite = np.isfinite(traj).all(axis=tuple(range(1, traj.ndim)))
    blowup = None if finite.all() else int(np.argmin(finite))
    return traj, al, blowup


def _alpha_layout(alphas, mesh, xL):
    """(x_axis, alphas_2d): per-element -> element centres; nodal (T,n_elem,P)
    -> interior interface midpoints, flattened."""
    if alphas.ndim == 3:
        xn = np.asarray(mesh.node_positions(xL))
        x = (0.5 * (xn[:, :-1] + xn[:, 1:])).ravel()
        return x, alphas.reshape(alphas.shape[0], -1)
    x = xL + (np.arange(mesh.n_elem) + 0.5) * mesh.dx
    return x, alphas


def _space_mean(alphas, end):
    return np.nanmean(alphas[:end].reshape(end, -1), axis=1)


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def load_trained_model(model_path):
    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    model = load_model(model_path, build_from_meta(meta, jax.random.PRNGKey(0)))
    return model, meta


def compare_schemes(cfg, case, seed, model_path, alpha_max=None, n_save=200,
                    n_steps_override=None, outdir="nn/evaluation/figures"):
    model, meta = load_trained_model(model_path)
    mtype = meta.get("model_type", "element")
    channels = channels_from_meta(meta)     # the input this checkpoint was trained on
    if alpha_max is None:
        alpha_max = float(meta.get("alpha_max", cfg.alpha_max))
    if int(meta.get("P", cfg.P)) != cfg.P:
        print(f"[warn] model P={meta.get('P')} != cfg P={cfg.P}")
    print(f"       data_channels={channels}  precondition={meta.get('precondition', False)}")

    cs = setup_case(case, cfg, seed)
    mesh, xL, xR, tag = cs["mesh"], cs["xL"], cs["xR"], cs["tag"]
    n_steps = n_steps_override or cs["n_steps"]
    stride = max(1, n_steps // n_save)
    n_out = n_steps // stride
    times = np.arange(n_out + 1) * stride * cfg.dt
    print(f"[{tag}] n_elem={cfg.n_elem} P={cfg.P} dt={cfg.dt:.1e} "
          f"steps={n_out*stride} (T={n_out*stride*cfg.dt:.4f}) model={mtype} "
          f"alpha_max={alpha_max}")

    # --- MUSCL fine reference on this case's domain/BCs ---------------------
    grid = MusclEulerGrid(cfg.N_muscl, xL, xR)
    grid.set_state(cs["muscl_state"](grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps,
                              bc=cs["muscl_bc"])
    muscl_traj = solver.trajectory(n_out, stride * cfg.muscl_substeps)
    factor = cfg.N_muscl // cfg.N_cost
    ref_cost = muscl_traj.reshape(n_out + 1, 3, cfg.N_cost, factor).mean(-1)
    x_muscl = grid.x

    # --- the three schemes --------------------------------------------------
    U0 = cs["U0_dg"]
    runs = {}
    for scheme in SCHEMES:
        fn = make_alpha_fn(scheme, mesh, cfg, model, mtype, alpha_max, channels)
        runs[scheme] = run_scheme(fn, U0, mesh, cfg.dt, n_out, stride)
        _, _, blow = runs[scheme]
        note = f"BLEW UP at t={times[blow]:.4f}" if blow is not None else "stable"
        print(f"  {SCHEME_STYLE[scheme]['label']:32s} {note}")

    # --- relative L2 density error on the shared cost grid ------------------
    proj_c = uniform_projector(cfg.P, cfg.n_elem, cfg.proj_pts_per_elem)
    ref_rho = ref_cost[:, 0]
    errors = {}
    for scheme, (traj, _, blow) in runs.items():
        end = blow if blow is not None else len(traj)
        rho = np.asarray(jax.vmap(proj_c)(jnp.asarray(traj[:end])))[:, 0]
        errors[scheme] = np.sqrt(
            np.sum((rho - ref_rho[:end]) ** 2, axis=1)
            / np.sum(ref_rho[:end] ** 2, axis=1))

    xc_zoom = x_muscl[np.argmax(np.abs(np.gradient(muscl_traj[-1, 0], x_muscl)))]

    os.makedirs(outdir, exist_ok=True)
    fig1 = _figure_comparison(tag, cfg, mesh, xL, xR, times, runs, errors,
                              x_muscl, muscl_traj, xc_zoom)
    p1 = os.path.join(outdir, f"{tag}_comparison.png")
    fig1.savefig(p1, dpi=160); plt.close(fig1)

    fig2 = _figure_alpha_spacetime(tag, cfg, mesh, xL, times, runs)
    p2 = os.path.join(outdir, f"{tag}_alpha_spacetime.png")
    fig2.savefig(p2, dpi=160); plt.close(fig2)

    p3 = os.path.join(outdir, f"{tag}_metrics.csv")
    _write_metrics(p3, times, runs, errors)
    print(f"figures -> {p1}\n           {p2}\nmetrics -> {p3}")
    return runs, errors


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _dg_density_on_plot_grid(cfg, traj_state):
    proj = uniform_projector(cfg.P, cfg.n_elem, PLOT_PTS)
    return np.asarray(proj(jnp.asarray(traj_state)))[0]


def _plot_solutions(ax, x_plot, cfg, runs, x_ref, ref_rho_final, times):
    ax.plot(x_ref, ref_rho_final, color=INK, lw=1.4, label="MUSCL reference")
    for scheme in SCHEMES:
        traj, _, blow = runs[scheme]
        st = SCHEME_STYLE[scheme]
        idx = len(traj) - 1 if blow is None else max(blow - 1, 0)
        label = st["label"] + (f"  (blew up, t={times[blow]:.3f})"
                               if blow is not None else "")
        ax.plot(x_plot, _dg_density_on_plot_grid(cfg, traj[idx]),
                color=st["color"], ls=st["ls"], lw=2.0, label=label)


def _figure_comparison(tag, cfg, mesh, xL, xR, times, runs, errors, x_muscl,
                       muscl_traj, xc_zoom):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{tag}: DGSEM stabilization vs MUSCL "
                 f"({cfg.n_elem} elements, P={cfg.P})", color=INK, fontsize=13)

    N_plot = cfg.n_elem * PLOT_PTS
    x_plot = xL + (np.arange(N_plot) + 0.5) * (xR - xL) / N_plot
    L = xR - xL
    ref_rho_final = muscl_traj[-1, 0]

    ax = axes[0, 0]
    _plot_solutions(ax, x_plot, cfg, runs, x_muscl, ref_rho_final, times)
    draw_elements(ax, xL, xR, cfg.n_elem)
    ax.set_xlabel("x"); ax.set_ylabel("density ρ")
    ax.set_title("(a) Density at final time"); ax.legend(loc="best")

    ax = axes[0, 1]
    half = 0.08 * L
    _plot_solutions(ax, x_plot, cfg, runs, x_muscl, ref_rho_final, times)
    draw_elements(ax, xL, xR, cfg.n_elem)
    ax.set_xlim(xc_zoom - half, xc_zoom + half)
    win = (x_muscl > xc_zoom - half) & (x_muscl < xc_zoom + half)
    lo, hi = ref_rho_final[win].min(), ref_rho_final[win].max()
    pad = 0.35 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + pad); ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("x"); ax.set_ylabel("density ρ")
    ax.set_title("(b) Zoom on the strongest gradient (oscillations)")

    ax = axes[1, 0]
    for scheme in SCHEMES:
        st = SCHEME_STYLE[scheme]
        e = errors[scheme][1:]
        t = times[1:len(e) + 1]
        ax.plot(t, np.maximum(e, 1e-16), color=st["color"], ls=st["ls"],
                label=st["label"])
        _, _, blow = runs[scheme]
        if blow is not None and len(e):
            ax.plot(t[-1], max(e[-1], 1e-16), "x", color=st["color"],
                    markersize=10, markeredgewidth=2.5)
            ax.annotate("blow-up", (t[-1], max(e[-1], 1e-16)),
                        textcoords="offset points", xytext=(6, 6),
                        color=st["color"], fontsize=9)
    ax.set_yscale("log"); ax.set_xlabel("t")
    ax.set_ylabel("relative L2 density error")
    ax.set_title("(c) Error vs MUSCL over time"); ax.legend(loc="best")

    ax = axes[1, 1]
    # Relative dissipation BUDGET: cumulative injected dissipation (running sum of
    # the space-mean alpha) as a PERCENTAGE of Persson-Peraire's cumulative
    # dissipation up to time t (PP = 100%). The endpoint is the whole-run budget:
    # 200% = the scheme spent twice PP's total dissipation, DG = 0%. Cumulative
    # (not the instantaneous ratio) so it is not dominated by the noisy blow-ups
    # where PP momentarily injects almost nothing.
    _, pp_alphas, pp_blow = runs["pp"]
    pp_end = pp_blow if pp_blow is not None else len(pp_alphas)
    pp_cum = np.cumsum(_space_mean(pp_alphas, pp_end))
    # start once PP has accrued >=2% of its budget: before that the denominator
    # is tiny and the ratio is a meaningless transient (a scheme that injects
    # before PP would read thousands of %).
    thresh = 2e-2 * pp_cum[-1] if pp_cum[-1] > 0 else np.inf
    rel_all = []
    for scheme in SCHEMES:
        _, alphas, blow = runs[scheme]
        st = SCHEME_STYLE[scheme]
        end = blow if blow is not None else len(alphas)
        m = min(end, pp_end)
        rel = 100.0 * np.cumsum(_space_mean(alphas, m)) / pp_cum[:m]
        rel = np.where(pp_cum[:m] > thresh, rel, np.nan)   # hide where PP ~ 0
        ax.plot(times[:m], rel, color=st["color"], ls=st["ls"], label=st["label"])
        rel_all.append(rel[np.isfinite(rel)])
    # robust y-limit: keep the settled band and the 200% mark readable, clip any
    # residual startup transient instead of letting it set the scale.
    finite = np.concatenate(rel_all) if rel_all else np.array([100.0])
    top = 1.15 * float(np.nanpercentile(finite, 90)) if finite.size else 200.0
    ax.set_ylim(0.0, max(200.0, top))
    ax.axhline(100.0, color=INK_2, lw=0.6, ls=":", zorder=0)   # PP baseline guide
    ax.set_xlabel("t"); ax.set_ylabel("cumulative dissipation vs PP  (%)")
    ax.set_title("(d) Relative dissipation budget (PP = 100%)")
    ax.legend(loc="best")

    for ax in axes.ravel():
        finish_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def _figure_alpha_spacetime(tag, cfg, mesh, xL, times, runs):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.suptitle(f"{tag}: where and when the schemes inject dissipation",
                 color=INK, fontsize=13)

    layouts = {s: _alpha_layout(runs[s][1], mesh, xL) for s in ("pp", "nn")}
    vmax = max(max(float(np.nanmax(a)) for _, a in layouts.values()), 1e-6)

    xR = xL + cfg.n_elem * mesh.dx
    pm = None
    for ax, scheme in zip(axes, ("pp", "nn")):
        x_a, alphas = layouts[scheme]
        pm = ax.pcolormesh(x_a, times, alphas, cmap=ALPHA_CMAP, vmin=0.0,
                           vmax=vmax, shading="nearest", rasterized=True)
        draw_elements(ax, xL, xR, cfg.n_elem, on_heatmap=True)
        ax.set_title(SCHEME_STYLE[scheme]["label"], fontsize=10)
        ax.set_xlabel("x"); ax.grid(False)
    axes[0].set_ylabel("t")
    cb = fig.colorbar(pm, ax=axes, pad=0.015)
    cb.set_label("blending factor α", color=INK_2)
    cb.outline.set_visible(False)
    return fig


def _write_metrics(path, times, runs, errors):
    rows = []
    for scheme in SCHEMES:
        _, alphas, blow = runs[scheme]
        e = errors[scheme]
        rows.append({
            "scheme": SCHEME_STYLE[scheme]["label"],
            "stable": blow is None,
            "blowup_time": "" if blow is None else f"{times[blow]:.5f}",
            "final_rel_L2_density": f"{e[-1]:.5e}" if len(e) else "nan",
            "max_rel_L2_density": f"{np.max(e):.5e}" if len(e) else "nan",
            "mean_alpha": f"{np.nanmean(alphas):.5f}",
            "max_alpha": f"{np.nanmax(alphas):.5f}",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print("\n  " + "  |  ".join(rows[0].keys()))
    for r in rows:
        print("  " + "  |  ".join(str(v) for v in r.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--P", type=int, default=4, help="Polynomial order of the config")
    ap.add_argument("--case", default="sod", choices=CASES)
    ap.add_argument("--seed", type=int, default=0,
                    help="IC seed (only used by --case random)")
    ap.add_argument("--model",
                    default="nn/training/checkpoints/alpha_model_best.eqx")
    ap.add_argument("--n-elem", type=int, default=None,
                    help="override cfg.n_elem (model generalizes across it)")
    ap.add_argument("--rollout-steps", type=int, default=None,
                    help="override the horizon (steps); default = case time")
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="deployment cap (default: the training value)")
    ap.add_argument("--n-save", type=int, default=200)
    ap.add_argument("--outdir", default="nn/evaluation/figures")
    args = ap.parse_args()

    cfg = TrainConfig(P=args.P)
    if args.n_elem is not None:
        cfg = replace(cfg, n_elem=args.n_elem)
    compare_schemes(cfg, args.case, args.seed, args.model, args.alpha_max,
                    args.n_save, args.rollout_steps, args.outdir)


if __name__ == "__main__":
    main()
