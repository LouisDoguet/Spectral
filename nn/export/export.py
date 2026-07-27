"""Export a trained JAX/equinox NodalAlphaModel to the C++ `.nnx` binary that
`lib/equinox/AlphaNet` loads.

The `.nnx` format is self-describing (little-endian):

    magic  "EQXN"                 (4 bytes)
    u32    version
    u32    P, width, kernel_size, depth, n_data_channels
    f64    alpha_max
    f64[]  weights in equinox leaf order, C-order:
             lift.weight (width, in_ch, k), lift.bias (width),
             per ResBlock: conv1.w (width,width,k), conv1.b (width),
                           conv2.w (width,width,k), conv2.b (width),
             proj_w (width), proj_b (scalar)
           with in_ch = n_data_channels + (P+1).

Architecture is read from the actual weight tensors (P from the meta), so this
works whatever n_data_channels the checkpoint was trained with.

Usage:
    python nn/export/export.py <checkpoint_dir>            # -> <dir>/alpha_model.nnx
    python nn/export/export.py path/to/alpha_model_best.eqx --out model.nnx
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import equinox as eqx

from network.model import load_model
from network.policy import build_from_meta

MAGIC = b"EQXN"
VERSION = 1


def resolve_paths(checkpoint):
    """checkpoint may be a dir (auto-find best/last + meta) or a .eqx file."""
    if os.path.isdir(checkpoint):
        meta = os.path.join(checkpoint, "model_meta.json")
        for name in ("alpha_model_best.eqx", "alpha_model_last.eqx"):
            model = os.path.join(checkpoint, name)
            if os.path.exists(model):
                return model, meta
        raise FileNotFoundError(f"no alpha_model_*.eqx in {checkpoint}")
    return checkpoint, os.path.join(os.path.dirname(checkpoint), "model_meta.json")


def export(model_path, meta_path, out_path):
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("model_type", "nodal") != "nodal":
        raise SystemExit("export.py only supports the nodal model (model_type='nodal')")
    if meta.get("precondition", False):
        raise SystemExit(
            "export.py: this checkpoint uses input preconditioning "
            "(model_meta 'precondition': true), which the C++ equinox runtime "
            "does not replicate. Retrain with precondition=False to export, or "
            "extend lib/equinox/AlphaNet::assembleFeatures to whiten the data "
            "channels the same way (see network.model.whiten_rows).")

    model = load_model(model_path, build_from_meta(meta, jax.random.PRNGKey(0)))
    leaves = [np.asarray(l, dtype=np.float64)
              for l in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]

    # architecture straight from the tensors (robust to stale meta fields)
    width, in_ch, k = leaves[0].shape        # lift.weight (width, in_ch, k)
    P = int(meta["P"])
    n_data = in_ch - (P + 1)
    depth = (len(leaves) - 4) // 4           # minus lift(2) + proj(2), 4 per block
    alpha_max = float(meta.get("alpha_max", 1.0))
    if len(leaves) != 2 + 4 * depth + 2:
        raise SystemExit(f"unexpected leaf count {len(leaves)} (P={P}, depth={depth})")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        for v in (P, width, k, depth, n_data):
            f.write(struct.pack("<I", int(v)))
        f.write(struct.pack("<d", alpha_max))
        for leaf in leaves:
            f.write(np.ascontiguousarray(leaf.ravel(), dtype="<f8").tobytes())

    print(f"exported {out_path}\n  P={P} width={width} kernel={k} depth={depth} "
          f"n_data_channels={n_data} in_ch={in_ch} alpha_max={alpha_max}  "
          f"({len(leaves)} tensors)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help="checkpoint directory or an .eqx file")
    ap.add_argument("--meta", default=None, help="override model_meta.json path")
    ap.add_argument("--out", default=None, help="output .nnx path")
    args = ap.parse_args()

    model_path, meta_path = resolve_paths(args.checkpoint)
    if args.meta:
        meta_path = args.meta
    out = args.out or os.path.join(os.path.dirname(model_path), "alpha_model.nnx")
    export(model_path, meta_path, out)


if __name__ == "__main__":
    main()
