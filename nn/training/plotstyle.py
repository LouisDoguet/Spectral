"""
plotstyle.py — Publication-ready matplotlib wrapper
----------------------------------------------------
Usage:
    import plotstyle  # applies styles globally on import
    # or explicitly:
    plotstyle.apply()

All standard matplotlib/pyplot usage works unchanged after importing.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from cycler import cycler

# ── Palette ────────────────────────────────────────────────────────────────
# Colour-blind-safe palette (Wong 2011)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/violet
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# ── Dimensions (inches) ────────────────────────────────────────────────────
# Common journal column widths:
#   single column: ~3.3–3.5"   double column: ~6.5–7"
SINGLE_COL = (3.5, 2.625)  # golden-ratio height
DOUBLE_COL = (7.0, 5.25)
WIDE       = (7.0, 3.5)    # wide & short (e.g. time series)

# ── Core settings ──────────────────────────────────────────────────────────
RC = {
    # --- Figure ---
    "figure.figsize":        SINGLE_COL,
    "figure.dpi":            150,          # screen preview
    "savefig.dpi":           600,          # print quality
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "figure.facecolor":      "white",
    "figure.edgecolor":      "white",

    # --- Font ---
    "font.family":           "serif",
    "font.size":             16,
    "axes.titlesize":        20,
    "axes.labelsize":        18,
    "xtick.labelsize":       16,
    "ytick.labelsize":       16,
    "legend.fontsize":       15,
    "legend.title_fontsize": 16,

    # --- Axes ---
    "axes.linewidth":        1.2,
    "axes.edgecolor":        "black",
    "axes.labelcolor":       "black",
    "axes.titlecolor":       "black",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "axes.axisbelow":        True,
    "axes.facecolor":        "white",
    "axes.prop_cycle":       cycler("color", PALETTE),

    # --- Grid ---
    "grid.color":            "#dddddd",
    "grid.linewidth":        0.5,
    "grid.linestyle":        "--",
    "grid.alpha":            0.7,

    # --- Lines & markers ---
    "lines.linewidth":       1.2,
    "lines.markersize":      4,
    "lines.markeredgewidth": 0.8,

    # --- Error bars ---
    "errorbar.capsize":      2,

    # --- Ticks ---
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.color":           "black",
    "ytick.color":           "black",
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
    "xtick.minor.width":     0.8,
    "ytick.minor.width":     0.8,
    "xtick.major.size":      6,
    "ytick.major.size":      6,
    "xtick.minor.size":      3,
    "ytick.minor.size":      3,
    "xtick.top":             False,
    "ytick.right":           False,

    # --- Legend ---
    "legend.frameon":        True,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#cccccc",
    "legend.fancybox":       False,

    # --- Math / LaTeX ---
    # Uses matplotlib's mathtext engine (no LaTeX install needed).
    # Switch to "text.usetex": True if you have a LaTeX distribution.
    "text.usetex":           False,
    "mathtext.fontset":      "stix",
    "font.serif":            ["STIXGeneral", "DejaVu Serif", "Times New Roman",
                              "Palatino", "Georgia", "serif"],
}


def apply(use_latex: bool = False, palette: list | None = None) -> None:
    """Apply publication-quality rcParams.

    Parameters
    ----------
    use_latex : bool
        If True, render text with a local LaTeX installation (produces the
        best math output but requires LaTeX + dvipng/dvisvgm).
    palette : list, optional
        Substitute colour palette; defaults to the Wong colour-blind-safe set.
    """
    rc = RC.copy()

    if use_latex:
        rc["text.usetex"] = True
        rc["font.family"] = "serif"  # LaTeX Computer Modern

    if palette is not None:
        rc["axes.prop_cycle"] = cycler("color", palette)

    mpl.rcParams.update(rc)


def figure(size: str | tuple = "single", **kwargs) -> plt.Figure:
    """Return a new Figure already sized for journals.

    Parameters
    ----------
    size : "single" | "double" | "wide" | (w, h)
        Preset column widths or an explicit (width, height) tuple in inches.
    """
    sizes = {"single": SINGLE_COL, "double": DOUBLE_COL, "wide": WIDE}
    figsize = sizes.get(size, size) if isinstance(size, str) else size
    return plt.figure(figsize=figsize, **kwargs)


def despine(ax: mpl.axes.Axes | None = None, keep: tuple = ("left", "bottom")) -> None:
    """Remove all spines except those in *keep*."""
    if ax is None:
        ax = plt.gca()
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(side in keep)


def draw_elements(ax, xL: float, xR: float, n_elem: int, *,
                  color: str = "#8a8a8a", lw: float = 0.4, alpha: float = 0.35,
                  shade: bool = True, shade_alpha: float = 0.05,
                  on_heatmap: bool = False, zorder: float = 0.5) -> None:
    """Mark the spectral-element boundaries on a spatial (x) axis.

    Draws the ``n_elem + 1`` element interfaces as thin vertical hairlines and,
    when *shade* is set, tints alternate elements very lightly so the element
    partition of the mesh is legible without competing with the data. Use this
    on any figure whose x-axis is physical space (solution plots, alpha fields,
    alpha space-time heatmaps) to show where each spectral element sits.

    Parameters
    ----------
    ax : matplotlib axes
    xL, xR : float          domain endpoints
    n_elem : int            number of (uniform) spectral elements
    on_heatmap : bool       switch to light hairlines and no shading, for lines
                            drawn on top of a filled pcolormesh/imshow.
    """
    edges = np.linspace(xL, xR, n_elem + 1)
    if on_heatmap:
        # lines must sit ABOVE the filled QuadMesh (zorder 1), not behind it
        color, alpha, shade, zorder = "white", max(alpha, 0.35), False, max(zorder, 3.0)
    if n_elem > 40:                 # thin out when the mesh is fine
        alpha *= 0.6
        lw *= 0.8
    if shade:
        for k in range(0, n_elem, 2):
            ax.axvspan(edges[k], edges[k + 1], color=color, alpha=shade_alpha,
                       lw=0, zorder=zorder - 0.4)
    for xe in edges:
        ax.axvline(xe, color=color, lw=lw, alpha=alpha, zorder=zorder)


def save(path: str, fig: plt.Figure | None = None, **kwargs) -> None:
    """Save *fig* (or current figure) at 600 dpi with tight layout.

    Supported formats are inferred from the extension:
    pdf, svg, png, eps, tiff — all work fine.
    Prefer pdf/svg for vector output (no pixelation at any zoom).
    """
    fig = fig or plt.gcf()
    defaults = dict(dpi=600, bbox_inches="tight", facecolor="white")
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    print(f"Saved → {path}")


# Apply immediately on import
apply()


# ── Quick demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    x = np.linspace(0, 4 * np.pi, 200)

    # --- single column line plot ---
    fig, ax = plt.subplots()
    for k, label in enumerate(["sin(x)", "cos(x)", "sin(2x)"]):
        freq = [1, 1, 2][k]
        fn   = [np.sin, np.cos, np.sin][k]
        ax.plot(x, fn(freq * x), label=label)
    ax.set_xlabel("x (rad)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Trigonometric functions")
    ax.legend()
    save("demo_line.pdf")
    plt.show()

    # --- double column scatter with error bars ---
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    rng = np.random.default_rng(0)
    for i, ax in enumerate(axes):
        y  = rng.normal(i, 0.5, 30)
        ye = rng.uniform(0.05, 0.2, 30)
        ax.errorbar(np.arange(30), y, yerr=ye, fmt="o", label=f"Group {i+1}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.legend()
    axes[0].set_title("Condition A")
    axes[1].set_title("Condition B")
    fig.tight_layout()
    save("demo_scatter.pdf")
    plt.show()
