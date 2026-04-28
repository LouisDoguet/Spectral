"""Load binary snapshot files written by the C++ solver.

Expected binary layout (little-endian):
  int32   n_elem
  int32   P
  float64 t
  float64[n_total]  rho
  float64[n_total]  rhou
  float64[n_total]  e
where n_total = n_elem * (P + 1).
"""
import os
import glob
import numpy as np


def _is_valid(snap: dict) -> bool:
    """Return False if any field contains NaN or Inf (exploded simulation)."""
    for key in ("rho", "rhou", "enrg"):
        if not np.all(np.isfinite(snap[key])):
            return False
    return True


def load_snapshot(path: str) -> dict:
    """Return a dict with keys: rho, rhou, enrg, t, n_elem, P."""
    with open(path, "rb") as f:
        n_elem  = np.frombuffer(f.read(4), dtype=np.int32)[0]
        P       = np.frombuffer(f.read(4), dtype=np.int32)[0]
        t       = np.frombuffer(f.read(8), dtype=np.float64)[0]
        n_total = int(n_elem) * (int(P) + 1)
        rho     = np.frombuffer(f.read(8 * n_total), dtype=np.float64).copy()
        rhou    = np.frombuffer(f.read(8 * n_total), dtype=np.float64).copy()
        enrg    = np.frombuffer(f.read(8 * n_total), dtype=np.float64).copy()
    return {"rho": rho, "rhou": rhou, "enrg": enrg,
            "t": float(t), "n_elem": int(n_elem), "P": int(P)}


def load_dataset(directory: str, pattern: str = "*.bin") -> list[dict]:
    """Load all valid snapshots from a directory, sorted by filename.

    Snapshots that contain NaN or Inf (exploded simulation) are silently
    discarded with a warning.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    valid, skipped = [], []
    for p in paths:
        snap = load_snapshot(p)
        if _is_valid(snap):
            valid.append(snap)
        else:
            skipped.append(os.path.basename(p))

    if skipped:
        print(f"[loader] Skipped {len(skipped)} corrupted snapshot(s): {skipped}")
    if not valid:
        raise ValueError("All snapshots are corrupted. Re-run the C++ solver with a non-zero --eps.")
    return valid
