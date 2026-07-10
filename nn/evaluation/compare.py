"""Three-way stabilization comparison: pure DGSEM vs hybrid+Persson-Peraire
vs hybrid+trained-NN, against a fine reference.

The demonstration for each benchmark case:
  - "dg"  : alpha = 0 everywhere -> the raw high-order scheme, which
            oscillates at shocks and can blow up (the stability problem),
  - "pp"  : the built-in Persson-Peraire indicator (the hand-tuned fix),
  - "nn"  : the trained alpha policy in deployment mode (hard clip, cap,
            neighbour diffusion -- exactly what the C++ solver would run).

Outputs (into --outdir):
  <case>_comparison.png       solution + shock zoom + error growth + alpha budget
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.solver import rk4_step, cfl_time_step
from jax_dgsem.indicator import (modal_energy, persson_peraire_alpha,
                                 postprocess_alpha)
from network.model import AlphaModel, load_model
from training.cost import uniform_projector
from training.data_loader import (SOD, LAX, riemann_ic, shu_osher_ic,
                                  wall_states_from_ic, refine_ic,
                                  generate_reference_trajectory)
from vizstyle import (apply_style, finish_axes, SCHEME_STYLE, ALPHA_CMAP,
                      INK, INK_2)

SCHEMES = ("dg", "pp", "nn")
CASE_T = {"sod": 0.2, "lax": 0.16, "shu-osher": 1.8}


# ---------------------------------------------------------------------------
# Case setup and scheme runs
# ---------------------------------------------------------------------------

def setup_case(case: str, P: int, n_elem: int, delta: float = None):
    """Returns (mesh, U0, xL, T_default) with wall BCs from the IC endpoints
    (how the C++ cases configure bc::Wall).

    delta: tanh smoothing half-width of the initial jump (default 2*dx, like
    the C++ cases). Smaller values sharpen the discontinuity — useful to
    push the unstabilized DG scheme into an actual blow-up.
    """
    basis = GLLBasis(P)
    dummy = ("wall", np.array([1.0, 0.0, 2.5]))
    if case == "shu-osher":
        xL, xR = -5.0, 5.0
        mesh = Mesh1D(basis, n_elem, xL, xR, dummy, dummy)
        U0 = shu_osher_ic(mesh, xL, delta)
    else:
        spec = {"sod": SOD, "lax": LAX}[case]
        xL, xR = spec["domain"]
        mesh = Mesh1D(basis, n_elem, xL, xR, dummy, dummy)
        U0 = riemann_ic(mesh, xL, spec["left"], spec["right"], spec["x0"],
                        delta)
    bcL, bcR = wall_states_from_ic(U0)
    mesh = eqx.tree_at(lambda m: (m.bc_left_state, m.bc_right_state), mesh,
                       (bcL, bcR))
    return mesh, U0, xL, CASE_T[case]


def make_alpha_fn(scheme: str, mesh, model=None, alpha_max: float = 0.5):
    if scheme == "dg":
        return lambda U: jnp.zeros(mesh.n_elem)
    if scheme == "pp":
        return lambda U: persson_peraire_alpha(U, mesh, alpha_max=alpha_max)
    if scheme == "nn":
        def fn(U):
            raw = model(modal_energy(U, mesh).T)
            # deployment mode: hard clip + cap + neighbour diffusion,
            # identical to the C++ HybridDGSEM network branch
            return postprocess_alpha(raw, alpha_min=0.001,
                                     alpha_max=alpha_max, diffuse=True,
                                     hard_clip=True)
        return fn
    raise ValueError(scheme)


def run_scheme(alpha_fn, U0, mesh, dt, n_out: int, save_every: int):
    """Advance n_out*save_every RK4 steps, saving every save_every steps.

    Returns (traj (n_out+1, 3, n_elem, Nn), alphas (n_out+1, n_elem),
    blowup_index or None). A blown-up run keeps its NaN tail so the plots can
    show exactly when it dies.
    """

    @eqx.filter_jit
    def _run(U0, dt):
        def inner(U, _):
            return rk4_step(U, alpha_fn(U), dt, mesh), None

        def outer(U, _):
            U1, _ = jax.lax.scan(inner, U, None, length=save_every)
            return U1, (U1, alpha_fn(U1))

        _, (traj, alphas) = jax.lax.scan(outer, U0, None, length=n_out)
        return traj, alphas

    traj, alphas = _run(U0, jnp.asarray(dt))
    traj = np.concatenate([np.asarray(U0)[None], np.asarray(traj)])
    alphas = np.concatenate([np.asarray(alpha_fn(U0))[None],
                             np.asarray(alphas)])
    finite = np.isfinite(traj).all(axis=(1, 2, 3))
    blowup = None if finite.all() else int(np.argmin(finite))
    return traj, alphas, blowup


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def load_trained_model(model_path: str):
    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    template = AlphaModel(meta["in_channels"], meta["width"],
                          meta["kernel_size"], meta["depth"],
                          key=jax.random.PRNGKey(0))
    return load_model(model_path, template), meta


def compare_schemes(case: str, model_path: str, n_elem: int = 50, P: int = 5,
                    T: float = None, cfl: float = 0.3, refine: int = 4,
                    alpha_max: float = None, n_save: int = 200,
                    proj_pts: int = 12, delta: float = None,
                    outdir: str = "nn/evaluation/figures"):
    model, meta = load_trained_model(model_path)
    if alpha_max is None:
        alpha_max = float(meta.get("alpha_max", 0.5))

    mesh, U0, xL, T_default = setup_case(case, P, n_elem, delta)
    T = T or T_default
    dt = float(cfl_time_step(U0, mesh, cfl))
    n_steps = int(np.ceil(T / dt))
    save_every = max(1, n_steps // n_save)
    n_out = n_steps // save_every
    T_run = n_out * save_every * dt
    times = np.arange(n_out + 1) * save_every * dt
    print(f"[{case}] n_elem={n_elem} P={P} dt={dt:.3e} "
          f"steps={n_out * save_every} (T={T_run:.4f}) alpha_max={alpha_max}")

    # --- reference: PP hybrid on the refine-x finer mesh, dt/refine --------
    basis = GLLBasis(P)
    mesh_f = Mesh1D(basis, n_elem * refine, xL,
                    xL + mesh.dx * n_elem, ("wall", mesh.bc_left_state),
                    ("wall", mesh.bc_right_state))
    ref = generate_reference_trajectory(
        refine_ic(U0, P, refine), mesh_f, n_out,
        jnp.asarray(save_every * dt), refine * save_every, alpha_max=1.0)
    ref = np.asarray(ref)

    # --- the three schemes --------------------------------------------------
    runs = {}
    for scheme in SCHEMES:
        alpha_fn = make_alpha_fn(scheme, mesh, model, alpha_max)
        runs[scheme] = run_scheme(alpha_fn, U0, mesh, dt, n_out, save_every)
        traj, _, blowup = runs[scheme]
        note = f"BLEW UP at t={times[blowup]:.4f}" if blowup is not None \
            else "stable"
        print(f"  {SCHEME_STYLE[scheme]['label']:32s} {note}")

    # --- errors on the shared uniform grid ----------------------------------
    assert proj_pts % refine == 0
    proj_c = uniform_projector(P, n_elem, proj_pts)
    proj_f = uniform_projector(P, n_elem * refine, proj_pts // refine)
    ref_rho = np.asarray(jax.vmap(proj_f)(jnp.asarray(ref)))[:, 0]  # density
    errors = {}
    for scheme, (traj, _, blowup) in runs.items():
        end = blowup if blowup is not None else len(traj)
        rho = np.asarray(jax.vmap(proj_c)(jnp.asarray(traj[:end])))[:, 0]
        e = np.sqrt(np.sum((rho - ref_rho[:end]) ** 2, axis=1)
                    / np.sum(ref_rho[:end] ** 2, axis=1))
        errors[scheme] = e

    # zoom window center: steepest density gradient of the reference at the
    # final time, computed on the uniform grid (fine GLL x-coordinates repeat
    # at element interfaces, so a gradient there divides by zero)
    n_uni = ref_rho.shape[1]
    x_uni = xL + (np.arange(n_uni) + 0.5) * (mesh.dx * n_elem / n_uni)
    xc_zoom = x_uni[np.argmax(np.abs(np.gradient(ref_rho[-1], x_uni)))]

    # --- outputs -------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    fig1 = _figure_comparison(case, mesh, xL, times, runs, errors, ref,
                              xc_zoom, n_elem, P)
    p1 = os.path.join(outdir, f"{case}_comparison.png")
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    fig2 = _figure_alpha_spacetime(case, mesh, xL, times, runs, alpha_max)
    p2 = os.path.join(outdir, f"{case}_alpha_spacetime.png")
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    p3 = os.path.join(outdir, f"{case}_metrics.csv")
    _write_metrics(p3, times, runs, errors)
    print(f"figures -> {p1}\n           {p2}\nmetrics -> {p3}")
    return runs, errors


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_solutions(ax, x_plot, runs, ref_x, ref_rho_final, times):
    ax.plot(ref_x, ref_rho_final, color=INK, lw=1.4,
            label="Reference (fine mesh)")
    for scheme in SCHEMES:
        traj, _, blowup = runs[scheme]
        st = SCHEME_STYLE[scheme]
        idx = len(traj) - 1 if blowup is None else max(blowup - 1, 0)
        label = st["label"]
        if blowup is not None:
            label += f"  (blew up, t={times[blowup]:.3f})"
        ax.plot(x_plot, traj[idx][0].ravel(), color=st["color"], ls=st["ls"],
                lw=2.0, label=label)


def _figure_comparison(case, mesh, xL, times, runs, errors, ref,
                       xc_zoom, n_elem, P):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{case}: DGSEM stabilization comparison "
                 f"({n_elem} elements, P={P})", color=INK, fontsize=13)

    x_plot = np.asarray(mesh.node_positions(xL)).ravel()
    L = mesh.dx * mesh.n_elem
    # fine-mesh node positions for the reference curve
    from jax_dgsem.basis import gll_nodes
    n_f = ref.shape[2]
    dx_f = L / n_f
    q = gll_nodes(P)
    x_ref = (xL + dx_f * np.arange(n_f)[:, None]
             + (q[None, :] + 1.0) * dx_f / 2.0).ravel()
    ref_rho_final = ref[-1][0].ravel()

    # (a) final density, full domain ---------------------------------------
    ax = axes[0, 0]
    _plot_solutions(ax, x_plot, runs, x_ref, ref_rho_final, times)
    ax.set_xlabel("x")
    ax.set_ylabel("density ρ")
    ax.set_title("(a) Density at final time")
    ax.legend(loc="best")

    # (b) zoom on the steepest feature of the reference ---------------------
    ax = axes[0, 1]
    half = 0.08 * L
    _plot_solutions(ax, x_plot, runs, x_ref, ref_rho_final, times)
    ax.set_xlim(xc_zoom - half, xc_zoom + half)
    in_win = (x_ref > xc_zoom - half) & (x_ref < xc_zoom + half)
    lo, hi = ref_rho_final[in_win].min(), ref_rho_final[in_win].max()
    pad = 0.35 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("x")
    ax.set_ylabel("density ρ")
    ax.set_title("(b) Zoom on the strongest gradient (oscillations)")

    # (c) error growth -------------------------------------------------------
    ax = axes[1, 0]
    for scheme in SCHEMES:
        st = SCHEME_STYLE[scheme]
        e = errors[scheme][1:]      # skip t=0: identical ICs, error is 0
        t = times[1:len(e) + 1]
        ax.plot(t, np.maximum(e, 1e-16), color=st["color"], ls=st["ls"],
                label=st["label"])
        _, _, blowup = runs[scheme]
        if blowup is not None and len(e) > 0:
            ax.plot(t[-1], max(e[-1], 1e-16), "x", color=st["color"],
                    markersize=10, markeredgewidth=2.5)
            ax.annotate("blow-up", (t[-1], max(e[-1], 1e-16)),
                        textcoords="offset points", xytext=(6, 6),
                        color=st["color"], fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2 density error")
    ax.set_title("(c) Error vs reference over time")
    ax.legend(loc="best")

    # (d) dissipation budget --------------------------------------------------
    ax = axes[1, 1]
    for scheme in SCHEMES:
        traj, alphas, blowup = runs[scheme]
        st = SCHEME_STYLE[scheme]
        end = blowup if blowup is not None else len(alphas)
        ax.plot(times[:end], np.nanmean(alphas[:end], axis=1),
                color=st["color"], ls=st["ls"], label=st["label"])
    ax.set_xlabel("t")
    ax.set_ylabel("mean blending factor ᾱ(t)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("(d) Dissipation budget (lower = closer to pure DG)")
    ax.legend(loc="best")

    for ax in axes.ravel():
        finish_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def _figure_alpha_spacetime(case, mesh, xL, times, runs, alpha_max):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.suptitle(f"{case}: where and when the schemes inject dissipation",
                 color=INK, fontsize=13)

    L = mesh.dx * mesh.n_elem
    x_centers = xL + (np.arange(mesh.n_elem) + 0.5) * mesh.dx
    vmax = max(float(np.nanmax(runs[s][1])) for s in ("pp", "nn"))
    vmax = max(vmax, 1e-6)

    for ax, scheme in zip(axes, ("pp", "nn")):
        _, alphas, _ = runs[scheme]
        pm = ax.pcolormesh(x_centers, times, alphas, cmap=ALPHA_CMAP,
                           vmin=0.0, vmax=vmax, shading="nearest",
                           rasterized=True)
        ax.set_title(SCHEME_STYLE[scheme]["label"], fontsize=10)
        ax.set_xlabel("x")
        ax.grid(False)
    axes[0].set_ylabel("t")
    cb = fig.colorbar(pm, ax=axes, pad=0.015)
    cb.set_label("blending factor α", color=INK_2)
    cb.outline.set_visible(False)
    return fig


def _write_metrics(path, times, runs, errors):
    rows = []
    for scheme in SCHEMES:
        traj, alphas, blowup = runs[scheme]
        e = errors[scheme]
        rows.append({
            "scheme": SCHEME_STYLE[scheme]["label"],
            "stable": blowup is None,
            "blowup_time": "" if blowup is None else f"{times[blowup]:.5f}",
            "final_rel_L2_density": f"{e[-1]:.5e}" if len(e) else "nan",
            "max_rel_L2_density": f"{np.max(e):.5e}" if len(e) else "nan",
            "mean_alpha": f"{np.nanmean(alphas):.5f}",
            "max_alpha": f"{np.nanmax(alphas):.5f}",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n  " + "  |  ".join(rows[0].keys()))
    for r in rows:
        print("  " + "  |  ".join(str(v) for v in r.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="sod",
                    choices=["sod", "lax", "shu-osher"])
    ap.add_argument("--model",
                    default="nn/training/checkpoints/alpha_model_best.eqx")
    ap.add_argument("--n-elem", type=int, default=50)
    ap.add_argument("--P", type=int, default=5)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="deployment cap (default: the training value)")
    ap.add_argument("--n-save", type=int, default=200)
    ap.add_argument("--delta", type=float, default=None,
                    help="IC smoothing half-width (default 2*dx; smaller "
                         "= sharper jump, pushes the raw DG toward blow-up)")
    ap.add_argument("--outdir", default="nn/evaluation/figures")
    args = ap.parse_args()
    compare_schemes(args.case, args.model, args.n_elem, args.P, args.T,
                    args.cfl, args.refine, args.alpha_max, args.n_save,
                    delta=args.delta, outdir=args.outdir)


if __name__ == "__main__":
    main()
