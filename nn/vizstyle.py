"""Shared chart style for the training-analysis and scheme-comparison figures.

Categorical palette validated (CVD-safe, white surface #ffffff):
worst adjacent pair deltaE 47.2. Aqua and yellow sit below 3:1 contrast on the
white surface, so every figure pairs color with a second encoding (line style
+ direct labels / legend), never color alone.

Color follows the entity everywhere: a scheme keeps its color in every figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- categorical slots (fixed order, never cycled) -------------------------
BLUE = "#2a78d6"      # slot 1
AQUA = "#1baf7a"      # slot 2
YELLOW = "#eda100"    # slot 3
VIOLET = "#4a3aa7"    # slot 5 (4th series where needed)

# scheme -> (color, linestyle): the identity assignment used in EVERY figure
SCHEME_STYLE = {
    "dg": dict(color=BLUE, ls=(0, (5, 2)), label="DGSEM (no stabilization)"),
    "pp": dict(color=AQUA, ls=(0, (1, 1)), label="Hybrid + Persson-Peraire"),
    "nn": dict(color=YELLOW, ls="-", label="Hybrid + NN policy"),
}

# --- ink & chrome -----------------------------------------------------------
SURFACE = "#ffffff"
INK = "#000000"          # primary ink: reference curves, titles
INK_2 = "#000000"        # secondary ink: axis labels
MUTED = "#000000"        # tick labels
GRID = "#e1e0d9"         # hairline gridlines
BASELINE = "#000000"

# --- sequential ramp (single hue, light -> dark) for the alpha heatmaps ----
_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]
ALPHA_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)


def apply_style():
    """Idempotent, order-independent: pins every rcParam this house style
    cares about, so a stray global mutation from another module imported
    earlier (e.g. training.plotstyle's import-time apply(), which sets a
    dashed grid + a STIX mathtext font that clashes with the sans-serif body
    font) never leaks into a figure that calls this afterward."""
    plt.rcParams.update({
        "font.size": 15,
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 19,
        "legend.frameon": False,
        "legend.fontsize": 15,
        "lines.linewidth": 2.4,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",   # matches the sans-serif body font
        "text.usetex": False,
    })


def finish_axes(ax):
    """Recessive spines: keep left/bottom hairlines only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
