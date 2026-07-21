"""JAX-side reference for the export parity check.

Runs the trained NodalAlphaModel on the sharp Sod state on [0,1], N=32, then
diffs against the C++ dump produced by `build/equinox_parity` on the identical
state. Confirms the exported `.nnx` + C++ equinox runtime reproduce the JAX model
to machine precision.

    # 1. export the checkpoint
    python nn/export/export.py nn/training/checkpoints_pretrained_P4
    # 2. build + run the C++ harness (dumps state, alpha, feature channels)
    cmake --build build --target equinox_parity
    build/equinox_parity nn/training/checkpoints_pretrained_P4/alpha_model.nnx /tmp/cpp_parity.txt
    # 3. compare
    python nn/export/parity_check.py nn/training/checkpoints_pretrained_P4 /tmp/cpp_parity.txt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from jax_dgsem.basis import GLLBasis
from jax_dgsem.solver import Mesh1D
from jax_dgsem.physics import GAMMA
from network.model import load_model
from network.policy import build_from_meta, alpha_features, channel_residual, channel_energy

N_ELEM, XL, XR, X0 = 32, 0.0, 1.0, 0.5


def sod_state(mesh):
    """Sharp Sod initial state on the mesh GLL nodes (matches equinox_parity.cpp)."""
    x = np.asarray(mesh.node_positions(XL))
    rho = np.where(x < X0, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < X0, 1.0, 0.1)
    E = p / (GAMMA - 1.0) + 0.5 * rho * u**2
    return jnp.asarray(np.stack([rho, rho * u, E]))


def parse_cpp(path):
    """Parse the labelled blocks written by equinox_parity.cpp."""
    lines = open(path).read().splitlines()
    out, i = {}, 0
    while i < len(lines):
        head = lines[i].split()
        tag = head[0]
        if tag == "STATE":
            N, Nn = int(head[1]), int(head[2])
            vals = np.array([[float(v) for v in lines[i + 1 + j].split()]
                             for j in range(N * Nn)])
            out["state"] = vals.reshape(N, Nn, 3).transpose(2, 0, 1)  # (3,N,Nn)
            i += 1 + N * Nn
        elif tag == "ALPHA":
            N, P = int(head[1]), int(head[2])
            out["alpha"] = np.array([float(lines[i + 1 + k])
                                     for k in range(N * P)]).reshape(N, P)
            i += 1 + N * P
        elif tag == "RES":
            N, Nn = int(head[1]), int(head[2])
            out["res"] = np.array([float(lines[i + 1 + k])
                                   for k in range(N * Nn)]).reshape(N, Nn)
            i += 1 + N * Nn
        elif tag == "ENERGY":
            N = int(head[1])
            out["energy"] = np.array([float(lines[i + 1 + k]) for k in range(N)])
            i += 1 + N
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="checkpoint directory (dir with model_meta.json)")
    ap.add_argument("cpp_dump", help="output file written by build/equinox_parity")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.checkpoint, "model_meta.json")))
    P = meta["P"]
    basis = GLLBasis(P)
    mesh = Mesh1D(basis, N_ELEM, XL, XR)
    U = sod_state(mesh)

    model = load_model(os.path.join(args.checkpoint, "alpha_model_best.eqx"),
                       build_from_meta(meta, jax.random.PRNGKey(0)))
    alpha_j = np.asarray(model(alpha_features(U, mesh, "nodal")))     # (N, P)
    res_j = np.asarray(channel_residual(U, mesh))                    # (N, Nn)
    en_j = np.asarray(channel_energy(U, mesh))[:, 0]                 # (N,)

    c = parse_cpp(args.cpp_dump)

    def report(name, a, b):
        d = float(np.max(np.abs(a - b)))
        print(f"  {name:16s} max|cpp-jax| = {d:.3e}   {'PASS' if d < 1e-12 else 'FAIL'}")
        return d < 1e-12

    print(f"Parity: {args.checkpoint}  (P={P}, N={N_ELEM}, sharp Sod on [0,1])")
    ok = True
    ok &= report("state", c["state"], np.asarray(U))
    ok &= report("residual chan", c["res"], res_j)
    ok &= report("energy chan", c["energy"], en_j)
    ok &= report("alpha", c["alpha"], alpha_j)
    print(f"  alpha range: cpp [{c['alpha'].min():.4f}, {c['alpha'].max():.4f}]  "
          f"jax [{alpha_j.min():.4f}, {alpha_j.max():.4f}]")
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
