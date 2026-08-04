"""Summarize a P x n_elem sweep of compare.py runs into grid figures.

compare.py demonstrates the NN policy on ONE (P, n_elem, case) triple at a
time. This script reads the "*_metrics.csv" it writes across a whole sweep
(see SlurmJobs/compare_opno_grid.job) and renders, per case, a P x n_elem
heatmap -- the thing that actually shows P-/n_elem-independence, which no
single compare.py run can show on its own.

Expects results laid out as:
    <root>/P{P}_N{n_elem}/{case}_metrics.csv     (written by compare.py)

Run from the repo root, e.g.:
    .venv_spectral/bin/python nn/evaluation/aggregate_grid.py \\
        --root nn/compare/OPNO_C16 --P 4 6 8 --n-elem 32 64 96
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from vizstyle import apply_style, INK, INK_2, MUTED

CASES = ("sod", "lax", "shu-osher", "random_seed42")
CASE_TITLE = {"sod": "Sod", "lax": "Lax", "shu-osher": "Shu-Osher",
             "random_seed42": "Random IC (seed 42)"}
NN_ROW = "Hybrid + NN policy"
PP_ROW = "Hybrid + Persson-Peraire"

# single hue, light -> dark, for the absolute-error grid (mirrors
# vizstyle.ALPHA_CMAP's ramp style, a distinct hue from the alpha heatmaps)
ERROR_CMAP = LinearSegmentedColormap.from_list("seq_blue2", [
    "#dbe9fb", "#b7d3f6", "#8db8ef", "#5f9be6", "#3987e5", "#2a78d6",
    "#1c5cab", "#104281", "#0a2f5e"])
# diverging green->red around 100%, for the NN-vs-PP ratio grid
RATIO_CMAP = LinearSegmentedColormap.from_list("diverge_ratio", [
    "#0c5237", "#4fb489", "#eaf5ef", "#e4695b", "#6b1913"])


def read_grid(root, Ps, n_elems, cases):
    """-> list of dict rows: P, n_elem, case, scheme, stable, final_rel_L2_density,
    mean_alpha, max_alpha. Missing files are skipped with a warning (a
    partially-finished sweep should still summarize what's there)."""
    rows = []
    for P in Ps:
        for n in n_elems:
            for case in cases:
                path = os.path.join(root, f"P{P}_N{n}", f"{case}_metrics.csv")
                if not os.path.exists(path):
                    print(f"[warn] missing {path}, skipping")
                    continue
                with open(path) as f:
                    for r in csv.DictReader(f):
                        rows.append(dict(P=P, n_elem=n, case=case, **r))
    return rows


def _grid_values(rows, P, n, case, scheme_row, field):
    for r in rows:
        if r["P"] == P and r["n_elem"] == n and r["case"] == case \
                and r["scheme"] == scheme_row:
            return r
    return None


def _heatmap_panel(ax, Ps, n_elems, values, stable_mask, cmap, vmin, vmax,
                   fmt="{:.2e}", cbar_label=None, fig=None):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(n_elems)), [str(n) for n in n_elems])
    ax.set_yticks(range(len(Ps)), [f"P={p}" for p in Ps])
    for i in range(len(Ps)):
        for j in range(len(n_elems)):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8,
                        color=MUTED)
                continue
            unstable = not stable_mask[i, j]
            txt = fmt.format(v) + (" ✗" if unstable else "")
            color = INK if cmap is not None and (vmax - vmin) > 0 and \
                (v - vmin) / (vmax - vmin) < 0.55 else "#fcfcfb"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="#c23a2c" if unstable else color, fontweight="bold" if unstable else None)
    if fig is not None and cbar_label is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cb.set_label(cbar_label, color=INK_2, fontsize=8)
        cb.outline.set_visible(False)
    return im


def plot_error_grid(rows, Ps, n_elems, cases, title_suffix=""):
    """Final relative L2 density error of the NN policy, one heatmap per
    case, P x n_elem. This is the direct P-/n_elem-independence check: a
    flexible model should show roughly FLAT color across every panel, not a
    gradient with P or n_elem."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"OPNO policy: final relative L2 density error{title_suffix}\n"
                 "(flat color across a panel = P-/n_elem-independent)",
                 color=INK, fontsize=12)
    finite_vals = [float(r["final_rel_L2_density"]) for r in rows
                  if r["scheme"] == NN_ROW]
    vmin, vmax = (min(finite_vals), max(finite_vals)) if finite_vals else (0, 1)
    for ax, case in zip(axes.ravel(), cases):
        values = np.full((len(Ps), len(n_elems)), np.nan)
        stable = np.ones((len(Ps), len(n_elems)), dtype=bool)
        for i, P in enumerate(Ps):
            for j, n in enumerate(n_elems):
                r = _grid_values(rows, P, n, case, NN_ROW,
                                 "final_rel_L2_density")
                if r is None:
                    continue
                values[i, j] = float(r["final_rel_L2_density"])
                stable[i, j] = r["stable"] in ("True", "true", "1", True)
        _heatmap_panel(ax, Ps, n_elems, values, stable, ERROR_CMAP,
                      vmin, vmax, cbar_label="rel. L2 error", fig=fig)
        ax.set_xlabel("n_elem"); ax.set_ylabel("polynomial order")
        ax.set_title(CASE_TITLE.get(case, case), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_ratio_grid(rows, Ps, n_elems, cases, title_suffix=""):
    """NN error as a % of Persson-Peraire's error, same layout. <100% = NN
    beats the hand-tuned baseline at that (P, n_elem); the point is that this
    stays < 100% (green) everywhere in the grid, not just at the trained P."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"OPNO policy error as % of Persson-Peraire's error{title_suffix}\n"
                 "(< 100% = NN beats the hand-tuned baseline at that P, n_elem)",
                 color=INK, fontsize=12)
    for ax, case in zip(axes.ravel(), cases):
        values = np.full((len(Ps), len(n_elems)), np.nan)
        stable = np.ones((len(Ps), len(n_elems)), dtype=bool)
        for i, P in enumerate(Ps):
            for j, n in enumerate(n_elems):
                r_nn = _grid_values(rows, P, n, case, NN_ROW,
                                    "final_rel_L2_density")
                r_pp = _grid_values(rows, P, n, case, PP_ROW,
                                    "final_rel_L2_density")
                if r_nn is None or r_pp is None:
                    continue
                pp_err = float(r_pp["final_rel_L2_density"])
                nn_err = float(r_nn["final_rel_L2_density"])
                values[i, j] = 100.0 * nn_err / pp_err if pp_err > 0 else np.nan
                stable[i, j] = r_nn["stable"] in ("True", "true", "1", True)
        _heatmap_panel(ax, Ps, n_elems, values, stable, RATIO_CMAP, 0.0, 200.0,
                      fmt="{:.0f}%", cbar_label="NN / PP error  (%)", fig=fig)
        ax.set_xlabel("n_elem"); ax.set_ylabel("polynomial order")
        ax.set_title(CASE_TITLE.get(case, case), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def write_combined_csv(rows, path):
    if not rows:
        return
    fields = ["P", "n_elem", "case", "scheme", "stable", "blowup_time",
             "final_rel_L2_density", "max_rel_L2_density", "mean_alpha",
             "max_alpha"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def print_summary(rows, Ps, n_elems, cases):
    nn_rows = [r for r in rows if r["scheme"] == NN_ROW]
    n_expected = len(Ps) * len(n_elems) * len(cases)
    unstable = [r for r in nn_rows if r["stable"] not in ("True", "true", "1", True)]
    print(f"\n{len(nn_rows)}/{n_expected} (P, n_elem, case) cells found.")
    if unstable:
        print(f"NN policy UNSTABLE on {len(unstable)} cell(s):")
        for r in unstable:
            print(f"  P={r['P']} n_elem={r['n_elem']} case={r['case']} "
                 f"blew up at t={r['blowup_time']}")
    else:
        print("NN policy stable on every cell.")
    ratios = []
    for r in nn_rows:
        pp = _grid_values(rows, r["P"], r["n_elem"], r["case"], PP_ROW,
                          "final_rel_L2_density")
        if pp is not None and float(pp["final_rel_L2_density"]) > 0:
            ratios.append(float(r["final_rel_L2_density"]) /
                         float(pp["final_rel_L2_density"]))
    if ratios:
        print(f"NN/PP error ratio across the grid: mean={np.mean(ratios):.2f}  "
             f"min={np.min(ratios):.2f}  max={np.max(ratios):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="directory containing P{P}_N{n_elem}/ subfolders")
    ap.add_argument("--P", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--n-elem", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--outdir", default=None,
                    help="default: <root>/summary")
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.root, "summary")
    os.makedirs(outdir, exist_ok=True)

    rows = read_grid(args.root, args.P, args.n_elem, args.cases)
    if not rows:
        print(f"no metrics found under {args.root}; nothing to summarize")
        return

    write_combined_csv(rows, os.path.join(outdir, "grid_metrics.csv"))

    fig1 = plot_error_grid(rows, args.P, args.n_elem, args.cases,
                           args.title_suffix)
    p1 = os.path.join(outdir, "error_grid.png")
    fig1.savefig(p1, dpi=160); plt.close(fig1)

    fig2 = plot_ratio_grid(rows, args.P, args.n_elem, args.cases,
                           args.title_suffix)
    p2 = os.path.join(outdir, "error_ratio_vs_pp_grid.png")
    fig2.savefig(p2, dpi=160); plt.close(fig2)

    print(f"\nfigures -> {p1}\n           {p2}")
    print(f"combined metrics -> {os.path.join(outdir, 'grid_metrics.csv')}")
    print_summary(rows, args.P, args.n_elem, args.cases)


if __name__ == "__main__":
    main()
