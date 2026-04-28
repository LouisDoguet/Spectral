"""
ONNX inference helper.

Usage (command line):
  python infer.py av_model.onnx snapshot.bin

Usage (from code):
  session = load_model("av_model.onnx")
  eps = predict_eps(session, rho, rhou, enrg)
"""
import sys
import numpy as np
import onnxruntime as ort

from data.loader import load_snapshot


def load_model(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def predict_eps(session: ort.InferenceSession,
                rho: np.ndarray,
                rhou: np.ndarray,
                enrg: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    rho, rhou, enrg : (n_total,) float64 arrays (C++ solver output)

    Returns
    -------
    eps : (n_total,) float32 array with eps >= 0
    """
    state = np.concatenate([rho, rhou, enrg]).astype(np.float32)[np.newaxis, :]
    [eps] = session.run(["eps"], {"state": state})
    return eps[0]


if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "av_model.onnx"
    snap_path = sys.argv[2] if len(sys.argv) > 2 else None

    session = load_model(onnx_path)
    print(f"Loaded model from {onnx_path}")

    if snap_path:
        snap = load_snapshot(snap_path)
        eps  = predict_eps(session, snap["rho"], snap["rhou"], snap["enrg"])
        print(f"eps  min={eps.min():.4e}  max={eps.max():.4e}  mean={eps.mean():.4e}")
        out = snap_path.replace(".bin", "_eps.npy")
        np.save(out, eps)
        print(f"Saved → {out}")
