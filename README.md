# Spectral1D: High-Order Euler Solver

A 1D Spectral Element Method solver for the Euler equations.

- **High-Order Accuracy**: Uses Gauss-Lobatto-Legendre quadrature and Lagrange polynomials for spatial discretization.
- **Physics**: Implements the 1D Euler equations with a Rusanov Riemann solver for robust interface flux treatment.
- **Time Integration**: 4th-order Runge-Kutta method.
- **HPC Optimized**:
    - **Unified Buffer Strategy**: Conserved variables are stored in contiguous global arrays to maximize CPU cache locality.
    - **BLAS Integration**

## Code Structure
- `[EARLY DEV PHASE : RBF base switch]` `lib/base/`: GLL basis and derivative matrix construction. POssibility to define RBF basis and RBF solving 
- `lib/phy/`: Euler flux functions and Riemann solver. Entropy conserving and entropy stable flux calculation from Charnesav.
- `lib/math` : Math helpers
- `lib/space/`: `Mesh` and `Element` classes managing the unified memory and spatial operator.
- `lib/time/`: Optimized `RK4` class and data export routines. The hybrid solver blends the entropy-stable DG and FV *subcell fluxes* per interface (`(1-a)B_dg + a B_fv`), matching `nn/jax_dgsem`, so a per-interface (nodal) blending factor stays conservative and entropy-stable.
- `lib/equinox/`: inference-only C++ runtime for the trained JAX/equinox alpha policy (`Conv1d`, `ResBlock`, `AlphaNet`). Loads a `.nnx` file produced by `nn/export/export.py` and reproduces the JAX `NodalAlphaModel` to machine precision.

## Neural alpha policy: JAX → C++ export

The blending factor `alpha` can be driven either by the Persson-Peraire modal
indicator (default) or by a network trained in JAX. The network is trained,
exported, and consumed by the C++ solver as a single attachable object
(`HybridDGSEM::setAlphaNet`):

```
train (JAX, nn/) -> nn/export/export.py -> model.nnx -> build/spectral --nn_model model.nnx
```

```bash
python nn/export/export.py nn/training/checkpoints_pretrained_P4
build/spectral --solver hybrid_dgsem --case sod \
    --nn_model nn/training/checkpoints_pretrained_P4/alpha_model.nnx
```

Omit `--nn_model` to fall back to the Persson-Peraire indicator. Re-run
`export.py` after each training run (a stale `.nnx` no longer matches the
checkpoint). See [NN_EXPORT_MIGRATION.md](NN_EXPORT_MIGRATION.md) for the `.nnx`
format and the JAX↔C++ parity check (`nn/export/parity_check.py`).

## Hybrid Flux Blending FV/DGSEM solver
`JAX`-differentiable solver for CNN backpropagation.

<img src="./img/hDGSEM.drawio.png" width="250" alt="hDGSEM">

## `JAX` Neural Network

<img src="./img/NN.drawio.png" width="600" alt="JAX Neural Network">

Python CNN framework to learn the behavior of the blending coefficient.
- `nn/jax_dgsem` : `JAX`-differentiable DGSEM solver (previous section)
- `nn/muscl` : Python (non `JAX`-differentiable solver) diffusive solver, to generate reference solution (on a finer mesh). The solver is second-order accurate MUSCL scheme, with a `minmod` limiter to ensure stability of the solution, despite the really high resolution of the grid.
- `nn/network` : Networks structure. Element networks and Quadrature networks.
- `nn/training` : Training policy of the network.

<img src="./img/Training.drawio.png" width="600" alt="Training diagram">

## Build & Compile
Requires `BLAS`, `LAPACK`, `BOOST_PROGRAM_OPTIONS`
```bash
mkdir build
cd build
../cmake
make
```
