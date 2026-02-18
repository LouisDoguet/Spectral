# Spectral1D Project Context

## Overview
A 1D Spectral Element Method (SEM) solver for the Euler equations using Gauss-Lobatto-Legendre (GLL) quadrature and Riemann solvers for interface fluxes.

## Architecture
- **Basis (`lib/base/gll.h/cpp`)**: Handles GLL quadrature points, weights, and the derivative matrix $D_{ij}$.
- **Physics (`lib/phy/physics.h/cpp`)**: Implements Euler flux functions, pressure calculation (Perfect Gas Law, $\gamma=1.4$), and Rusanov (local Lax-Friedrichs) Riemann solver.
- **Space (`lib/space/`)**:
  - `Element`: Manages local conserved variables ($ho, ho u, E$), fluxes, and divergence calculations using BLAS (`dgemv`).
  - `Mesh`: Handles element connectivity, Riemann interface corrections, and boundary conditions.

## Key Fixes & Design Decisions
- **Newton Iteration (`gll.cpp`)**: Fixed a bug where quadrature points were not updating during Newton refinement.
- **Euler Fluxes (`physics.cpp`)**: Corrected momentum flux $F_2 = \frac{(ho u)^2}{ho} + p$.
- **Rusanov Flux (`physics.cpp`)**: Corrected numerical flux formula to $F^* = 0.5(F_L + F_R) - 0.5\lambda(U_R - U_L)$.
- **Surface Corrections (`mesh.cpp`)**: Applied the strong-form scaling factor of $\frac{1}{w_i J}$ for interface and boundary corrections.
- **Divergence Correction (`element.h`)**: Fixed `correctDivF*` to update the divergence array `divF` instead of the internal flux array `F`.

## Build & Test
- **Dependencies**: Requires a BLAS implementation (prefer `openblas`).
- **Test Suite**: `test_suite.cpp` validates mesh generation, flux physics, derivative accuracy, and boundary coupling.
- **Compile Command**:
  ```bash
  g++ -o test_suite test_suite.cpp lib/base/gll.cpp lib/math/math.cpp lib/phy/physics.cpp lib/space/element.cpp lib/space/mesh.cpp -lopenblas
  ```

## Development Guidelines
- Always verify the derivative matrix accuracy with a linear function test after basis modifications.
- Surface corrections must use the $1/(wJ)$ scaling for the strong form of the SEM.
