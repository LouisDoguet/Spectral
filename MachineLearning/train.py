"""
Training pipeline:
  1. Load state snapshots exported by the C++ solver.
  2. Compute optimal-eps labels via per-element minimisation.
  3. Train the MLP.
  4. Export to ONNX.
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.loader import load_snapshot
from data.sod_exact import GAMMA
from labels.sem1d import make_geometry
from labels.optimal_eps import compute_labels
from models.mlp import ArtificialViscosityNet

# --------------------------------------------------------------------------
# Configuration — edit these to match your simulation setup
# --------------------------------------------------------------------------
SNAPSHOT_DIR = "../results/snapshots"
N_ELEM  = 50
P       = 5
XL, XR  = 0.0, 1.0
DT      = 1e-4
EPS_MAX = 0.5

EPOCHS  = 200
BATCH   = 32
LR      = 1e-3

ONNX_PATH  = "av_model.onnx"
CACHE_PATH = "dataset_cache.npz"   # pre-computed labels are cached here

# Sod tube boundary states
RHO_L, U_L, P_L = 1.0,   0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1
# --------------------------------------------------------------------------


def _bc(rho, u, p):
    e = p / (GAMMA - 1) + 0.5 * rho * u ** 2
    return (rho, rho * u, e)


def build_dataset(geom, bc_L, bc_R):
    """Load snapshots and compute labels (or restore from cache)."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data["X"], data["Y"]

    paths = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.bin")))
    if not paths:
        raise FileNotFoundError(f"No .bin snapshots found in {SNAPSHOT_DIR}")

    X_list, Y_list = [], []
    for i, path in enumerate(paths):
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(path)}", flush=True)
        snap = load_snapshot(path)
        rho, rhou, enrg = snap["rho"], snap["rhou"], snap["enrg"]

        x = np.concatenate([rho, rhou, enrg], dtype=np.float32)
        y = compute_labels(snap, DT, geom, bc_L, bc_R,
                           eps_max=EPS_MAX).astype(np.float32)
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    np.savez(CACHE_PATH, X=X, Y=Y)
    print(f"Dataset cached to {CACHE_PATH}")
    return X, Y


def train_model(X_np, Y_np, n_total):
    X = torch.tensor(X_np)
    Y = torch.tensor(Y_np)

    loader  = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)
    model   = ArtificialViscosityNet(n_total)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for xb, yb in loader:
            pred  = model(xb)
            loss  = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(xb)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  MSE={total/len(X):.6f}")

    return model


def export_onnx(model, n_total, path):
    model.eval()
    dummy = torch.zeros(1, 3 * n_total)
    torch.onnx.export(
        model, dummy, path,
        input_names=["state"],
        output_names=["eps"],
        opset_version=17,
    )
    print(f"Model exported → {path}")


if __name__ == "__main__":
    geom    = make_geometry(N_ELEM, P, XL, XR)
    n_total = geom["n_total"]
    bc_L    = _bc(RHO_L, U_L, P_L)
    bc_R    = _bc(RHO_R, U_R, P_R)

    print("=== Building dataset ===")
    X, Y = build_dataset(geom, bc_L, bc_R)
    print(f"Samples: {len(X)}  input: {X.shape[1]}  output: {Y.shape[1]}")

    print("=== Training ===")
    model = train_model(X, Y, n_total)

    print("=== Exporting ===")
    export_onnx(model, n_total, ONNX_PATH)
