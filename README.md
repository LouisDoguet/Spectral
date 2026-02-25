# Spectral1D: High-Order Euler Solver

A 1D Spectral Element Method solver for the Euler equations.

- **High-Order Accuracy**: Uses Gauss-Lobatto-Legendre quadrature and Lagrange polynomials for spatial discretization.
- **Physics**: Implements the 1D Euler equations with a Rusanov Riemann solver for robust interface flux treatment.
- **Time Integration**: 4th-order Runge-Kutta method.
- **HPC Optimized**:
    - **Unified Buffer Strategy**: Conserved variables are stored in contiguous global arrays to maximize CPU cache locality.
    - **BLAS Integration**: 

## Code Structure
- `lib/base/`: GLL basis and derivative matrix construction.
- `lib/phy/`: Euler flux functions and Riemann solver.
- `lib/space/`: `Mesh` and `Element` classes managing the unified memory and spatial operator.
- `lib/time/`: Optimized `RK4` class and data export routines.

## Build & Compile
Requires BLAS and BOOST_PROGRAM_OPTIONS
```bash
mkdir build
cd build
cmake ../
make
```
Then the executable is `spectral`
