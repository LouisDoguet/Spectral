"""Shared chart style for the training-analysis and scheme-comparison figures.

Categorical palette validated (CVD-safe, light surface #fcfcfb):
worst adjacent pair deltaE 47.2. Aqua and yellow sit below 3:1 contrast on the
light surface, so every figure pairs color with a second encoding (line style
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
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary ink: reference curves, titles
INK_2 = "#52514e"        # secondary ink: axis labels
MUTED = "#898781"        # tick labels
GRID = "#e1e0d9"         # hairline gridlines
BASELINE = "#c3c2b7"

# --- sequential ramp (single hue, light -> dark) for the alpha heatmaps ----
_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]
ALPHA_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
    })


def finish_axes(ax):
    """Recessive spines: keep left/bottom hairlines only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
