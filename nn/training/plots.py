"""Training-analysis recap figure from training_history.npz.

Run standalone after (or during) a training:
    .venv_spectral/bin/python nn/training/plots.py [path/to/training_history.npz]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from vizstyle import (apply_style, finish_axes, SCHEME_STYLE, BLUE, AQUA,
                      YELLOW, VIOLET, INK, MUTED)


def plot_training_history(history, out_path: str = None):
    """history: path to training_history.npz or an already-loaded dict.

    Four panels:
      (a) train / validation loss per epoch (the proof the network learns),
      (b) per-batch training loss (within-epoch behaviour, epoch means overlaid),
      (c) weighted validation cost contributions (what the optimizer trades off),
      (d) alpha statistics on the validation set (what the policy converges to).
    """
    h = dict(np.load(history)) if isinstance(history, (str, os.PathLike)) \
        else dict(history)
    ep = h["epoch"]

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Alpha-policy training analysis", color=INK, fontsize=13)

    # (a) loss curves ------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(ep, h["train_loss"], color=BLUE, label="train")
    ax.plot(ep, h["val_loss"], color=AQUA, ls=(0, (1, 1)), label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss  J(θ)")
    ax.set_title("(a) Loss per epoch")
    ax.legend()

    # (b) per-batch loss ---------------------------------------------------
    ax = axes[0, 1]
    bl, be = h["batch_loss"], h["batch_epoch"]
    ax.plot(np.arange(len(bl)), bl, color=BLUE, lw=0.8, alpha=0.45)
    # epoch means at epoch centers, on the batch axis
    centers, means = [], []
    for e in np.unique(be):
        idx = np.where(be == e)[0]
        centers.append(idx.mean())
        means.append(np.mean(np.asarray(bl)[idx]))
    ax.plot(centers, means, color=BLUE, lw=2.0, label="epoch mean")
    ax.set_yscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("batch loss")
    ax.set_title("(b) Per-batch training loss")
    ax.legend()

    # (c) weighted cost contributions on validation ------------------------
    ax = axes[1, 0]
    contribs = [
        ("w_osc · C_osc", h["w_osc"] * h["val_c_osc"], BLUE, (0, (5, 2))),
        ("w_acc · C_acc", h["w_acc"] * h["val_c_acc"], AQUA, (0, (1, 1))),
        ("w_α · C_α", h["w_alpha"] * h["val_c_alpha"], YELLOW, "-"),
    ]
    for label, y, color, ls in contribs:
        if np.all(np.asarray(y) <= 0):
            continue        # a disabled term (weight 0) would break the log axis
        ax.plot(ep, y, color=color, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted contribution")
    ax.set_title("(c) Validation cost breakdown")
    ax.legend()

    # (d) alpha statistics --------------------------------------------------
    ax = axes[1, 1]
    ax.plot(ep, h["val_alpha_mean"], color=VIOLET, label="mean α")
    ax.plot(ep, h["val_alpha_max"], color=VIOLET, ls=(0, (5, 2)), label="max α")
    ax.set_xlabel("epoch")
    ax.set_ylabel("blending factor α")
    ax.set_ylim(bottom=0.0)
    ax.set_title("(d) Policy output on validation set")
    ax.legend()

    for ax in axes.ravel():
        finish_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path:
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path
    return fig


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "nn/training/checkpoints/training_history.npz"
    out = plot_training_history(path, os.path.join(os.path.dirname(path),
                                                   "training_recap.png"))
    print(f"wrote {out}")
