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

## Quick Start
1. **Compile**:
   ```bash
   g++ -o main main.cpp lib/base/gll.cpp lib/math/math.cpp lib/phy/physics.cpp lib/space/element.cpp lib/space/mesh.cpp lib/time/rk4.cpp -lopenblas
   ```
