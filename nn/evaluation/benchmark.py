"""Benchmark the alpha-policy network across hyperparameters.

Strategy: ONE-FACTOR-AT-A-TIME (OFAT). A full factorial of every axis is ~1700
trainings (infeasible); instead we fix a baseline and sweep each axis on its
own -- ~18 trainings that show what each knob does. Edit BASELINE / the sweeps
in make_configs() to widen or narrow the study.

Each config gets a self-describing name, e.g.
    P4_N32_M64_R512_K5_wo5_wa5
    (P=4, n_elem=32, muscl_cells=64, rollout=512, kernel=5, w_osc=1e-5, w_alpha=1e-5)
and trains into  nn/training/benchmark/<name>/  (model_meta.json records the
full config). Each trained model is then evaluated on the SAME four cases
(sod, lax, shu-osher, random) against the fine MUSCL reference, and one JSON
summary per model is written to nn/training/benchmark/results/<name>.json.

Usage (usually via SlurmJobs/benchmark.job):
    python nn/evaluation/benchmark.py --list                 # show configs + ids
    python nn/evaluation/benchmark.py --task-id 0 --epochs 80 # train+eval config 0
    python nn/evaluation/benchmark.py --aggregate            # build the ranked table
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from training.config import TrainConfig
from training.train import train
from training.cost import uniform_projector
from network.model import load_model
from network.policy import build_from_meta
from muscl.euler import MusclEulerGrid, MusclEulerSolver
from evaluation.compare import setup_case, make_alpha_fn, run_scheme, CASES

BENCH_DIR = "nn/training/benchmark"
RESULTS_DIR = os.path.join(BENCH_DIR, "results")

# Same evaluation for every model: fixed physical horizon per case (so configs
# with different dt are compared over the same physical time) and one fixed
# random IC seed shared by all models.
EVAL_T = {"sod": 0.2, "lax": 0.16, "shu-osher": 0.6, "random": 0.1}
EVAL_SEED = 12345
EVAL_NSAVE = 60

# The baseline every sweep varies one axis away from.
BASELINE = dict(P=4, n_elem=32, muscl_cells=64, rollout=512, kernel="P+1",
                w_osc=1e-5, w_acc=1.0, w_alpha=1e-5)


# ---------------------------------------------------------------------------
# Config construction and naming
# ---------------------------------------------------------------------------

def kernel_value(spec, P):
    """'3' -> 3, 'P+1' -> P+1, '2P+1' -> 2P+1 (all odd for even P)."""
    return {"3": 3, "P+1": P + 1, "2P+1": 2 * P + 1}[spec]


def _exp(w):
    """1e-5 -> 5 (for compact, powers-of-ten weight names)."""
    return int(round(-math.log10(w)))


def config_name(p, P):
    ks = kernel_value(p["kernel"], P)
    return (f"P{P}_N{p['n_elem']}_M{p['muscl_cells']}_R{p['rollout']}"
            f"_K{ks}_wo{_exp(p['w_osc'])}_wa{_exp(p['w_alpha'])}")


def make_config(p, epochs):
    """Full TrainConfig for one benchmark point (dt from the subcell-CFL rule)."""
    P, ne = p["P"], p["n_elem"]
    name = config_name(p, P)
    return name, TrainConfig(
        P=P, n_elem=ne, muscl_cells_per_elem=p["muscl_cells"],
        rollout_steps=p["rollout"], kernel_size=kernel_value(p["kernel"], P),
        dt=TrainConfig.stable_dt(P, ne),
        w_osc=p["w_osc"], w_acc=p["w_acc"], w_alpha=p["w_alpha"],
        epochs=epochs, plot_every=0, n_val=3,
        checkpoint_dir=os.path.join(BENCH_DIR, name))


def make_configs(epochs):
    """The OFAT sweep: baseline + one sweep per axis. Returns an ordered list
    of (name, cfg, swept_axis); duplicates of the baseline are dropped."""
    configs, order = {}, []

    def add(overrides, axis):
        p = {**BASELINE, **overrides}
        name, cfg = make_config(p, epochs)
        if name not in configs:
            configs[name] = (cfg, axis)
            order.append(name)

    add({}, "baseline")
    for P in (4, 6, 8):
        add({"P": P}, "P")
    for ne in (16, 32, 64, 128):
        add({"n_elem": ne}, "n_elem")
    for mc in (16, 32, 64, 128):
        add({"muscl_cells": mc}, "muscl_cells")
    for r in (128, 256, 512, 1024):
        add({"rollout": r}, "rollout")
    for k in ("3", "P+1", "2P+1"):
        add({"kernel": k}, "kernel")
    for wo in (1e-6, 1e-5, 1e-4):
        add({"w_osc": wo}, "w_osc")
    for wa in (1e-6, 1e-5, 1e-4):
        add({"w_alpha": wa}, "w_alpha")

    return [(n, *configs[n]) for n in order]


# ---------------------------------------------------------------------------
# Evaluation of one trained model on the four cases
# ---------------------------------------------------------------------------

def load_best_or_last(ckpt_dir):
    """Load the best checkpoint, or fall back to the last (best may be missing
    if validation never improved)."""
    with open(os.path.join(ckpt_dir, "model_meta.json")) as f:
        meta = json.load(f)
    for fname in ("alpha_model_best.eqx", "alpha_model_last.eqx"):
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            model = load_model(path, build_from_meta(meta, jax.random.PRNGKey(0)))
            return model, meta
    raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")


def evaluate_case(case, cfg, model, meta):
    """Run dg/pp/nn on one case vs the MUSCL reference; return per-scheme
    (stable, final relative-L2 density error, mean alpha). A blown-up run gets
    final_err = 1.0 so it ranks as maximally bad."""
    cs = setup_case(case, cfg, EVAL_SEED)
    mesh, xL, xR = cs["mesh"], cs["xL"], cs["xR"]
    n_steps = max(EVAL_NSAVE, round(EVAL_T[case] / cfg.dt))
    stride = max(1, n_steps // EVAL_NSAVE)
    n_out = n_steps // stride

    grid = MusclEulerGrid(cfg.N_muscl, xL, xR)
    grid.set_state(cs["muscl_state"](grid))
    solver = MusclEulerSolver(grid, dt=cfg.dt / cfg.muscl_substeps,
                              bc=cs["muscl_bc"])
    muscl = solver.trajectory(n_out, stride * cfg.muscl_substeps)
    block = cfg.N_muscl // cfg.N_cost
    ref_rho = muscl.reshape(n_out + 1, 3, cfg.N_cost, block).mean(-1)[:, 0]

    proj = uniform_projector(cfg.P, cfg.n_elem, cfg.proj_pts_per_elem)
    amax = float(meta.get("alpha_max", cfg.alpha_max))
    mtype = meta.get("model_type", "element")

    out = {}
    for scheme in ("dg", "pp", "nn"):
        fn = make_alpha_fn(scheme, mesh, cfg, model, mtype, amax)
        traj, alphas, blow = run_scheme(fn, cs["U0_dg"], mesh, cfg.dt, n_out,
                                        stride)
        end = blow if blow is not None else len(traj)
        rho = np.asarray(jax.vmap(proj)(jnp.asarray(traj[:end])))[:, 0]
        err = np.sqrt(np.sum((rho - ref_rho[:end]) ** 2, axis=1)
                      / np.sum(ref_rho[:end] ** 2, axis=1))
        out[scheme] = dict(stable=blow is None,
                           final_err=float(err[-1]) if blow is None else 1.0,
                           mean_alpha=float(np.nanmean(alphas)))
    return out


def evaluate_model(name, cfg, axis):
    model, meta = load_best_or_last(cfg.checkpoint_dir)
    per_case = {}
    for case in CASES:
        try:
            per_case[case] = evaluate_case(case, cfg, model, meta)
        except Exception as e:                      # one bad case != lose all
            per_case[case] = dict(error=str(e))

    nn_err = [per_case[c].get("nn", {}).get("final_err", 1.0) for c in CASES]
    n_stable = sum(per_case[c].get("nn", {}).get("stable", False) for c in CASES)
    beats_dg = sum(per_case[c].get("nn", {}).get("final_err", 1.0)
                   < per_case[c].get("dg", {}).get("final_err", 1.0)
                   for c in CASES)
    return dict(name=name, axis=axis, P=cfg.P, n_elem=cfg.n_elem,
                muscl_cells=cfg.muscl_cells_per_elem, rollout=cfg.rollout_steps,
                kernel=cfg.kernel_size, dt=cfg.dt, w_osc=cfg.w_osc,
                w_alpha=cfg.w_alpha, mean_nn_err=float(np.mean(nn_err)),
                n_stable=int(n_stable), beats_dg=int(beats_dg),
                per_case=per_case)


# ---------------------------------------------------------------------------
# Task runner (train one config, then evaluate it)
# ---------------------------------------------------------------------------

def run_task(task_id, epochs):
    configs = make_configs(epochs)
    if not 0 <= task_id < len(configs):
        raise SystemExit(f"task-id {task_id} out of range 0..{len(configs)-1}")
    name, cfg, axis = configs[task_id]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=== benchmark task {task_id}: {name}  (sweeping {axis}) ===")

    try:
        train(cfg)
        summary = evaluate_model(name, cfg, axis)
        summary["trained"] = True
    except Exception as e:
        summary = dict(name=name, axis=axis, trained=False, error=str(e))
        print(f"config {name} FAILED: {e}")

    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {os.path.join(RESULTS_DIR, name + '.json')}")

    # Refresh the ranked table from whatever results exist so far; the last
    # array task to finish leaves the complete table (no second command).
    try:
        aggregate()
    except Exception as e:
        print(f"(aggregate skipped: {e})")


# ---------------------------------------------------------------------------
# Aggregation into the ranked table
# ---------------------------------------------------------------------------

def aggregate():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"no result files in {RESULTS_DIR}")

    done = [r for r in rows if r.get("trained")]
    # rank: most cases stable first, then lowest mean NN error
    done.sort(key=lambda r: (-r["n_stable"], r["mean_nn_err"]))

    cols = ["rank", "name", "axis", "P", "n_elem", "muscl_cells", "rollout",
            "kernel", "w_osc", "w_alpha", "n_stable", "beats_dg", "mean_nn_err",
            "err_sod", "err_lax", "err_shu", "err_random"]

    def err(r, case):
        return r["per_case"].get(case, {}).get("nn", {}).get("final_err", float("nan"))

    def record(r, rank):
        return {"rank": rank, "name": r["name"], "axis": r["axis"], "P": r["P"],
                "n_elem": r["n_elem"], "muscl_cells": r["muscl_cells"],
                "rollout": r["rollout"], "kernel": r["kernel"],
                "w_osc": r["w_osc"], "w_alpha": r["w_alpha"],
                "n_stable": r["n_stable"], "beats_dg": r["beats_dg"],
                "mean_nn_err": r["mean_nn_err"], "err_sod": err(r, "sod"),
                "err_lax": err(r, "lax"), "err_shu": err(r, "shu-osher"),
                "err_random": err(r, "random")}

    table = [record(r, i + 1) for i, r in enumerate(done)]

    # CSV
    import csv
    csv_path = os.path.join(BENCH_DIR, "RANKING.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    # Markdown
    md_path = os.path.join(BENCH_DIR, "RANKING.md")
    with open(md_path, "w") as f:
        f.write("# Alpha-policy benchmark ranking\n\n")
        f.write(f"{len(done)} trained / {len(rows)} configs. Ranked by cases "
                "stable (of 4) then mean NN relative-L2 density error vs MUSCL "
                "(lower = better). Errors are per case; blow-up counts as 1.0.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in table:
            def fmt(v):
                return f"{v:.4g}" if isinstance(v, float) else str(v)
            f.write("| " + " | ".join(fmt(row[c]) for c in cols) + " |\n")
        failed = [r for r in rows if not r.get("trained")]
        if failed:
            f.write("\n**Did not train:** "
                    + ", ".join(r["name"] for r in failed) + "\n")

    print(f"ranking -> {csv_path}\n           {md_path}")
    if table:
        best = table[0]
        print(f"\nBEST: {best['name']}  "
              f"({best['n_stable']}/4 stable, mean NN err {best['mean_nn_err']:.4g})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print configs + ids")
    ap.add_argument("--count", action="store_true", help="print the number of configs")
    ap.add_argument("--task-id", type=int, default=None, help="train+eval one config")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--aggregate", action="store_true", help="build the ranked table")
    args = ap.parse_args()

    if args.count:
        print(len(make_configs(args.epochs)))
    elif args.list:
        for i, (name, cfg, axis) in enumerate(make_configs(args.epochs)):
            print(f"{i:3d}  {name:36s} sweep={axis:12s} "
                  f"dt={cfg.dt:.2e} rollout={cfg.rollout_steps}")
    elif args.aggregate:
        aggregate()
    elif args.task_id is not None:
        run_task(args.task_id, args.epochs)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
