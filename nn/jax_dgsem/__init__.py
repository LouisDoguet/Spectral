"""Differentiable JAX replica of the C++ entropy-stable hybrid DGSEM solver.

This package exists so the training loss can be differentiated through the
scheme (differentiable-physics training of the alpha blending policy). It is
NOT a production solver: no I/O, no adaptivity, fixed dt, 1D Euler only.

The math mirrors lib/ exactly:
  - basis.py     <-> lib/base/gll.cpp, lib/math/math.cpp
  - physics.py   <-> lib/phy/physics.cpp, lib/phy/entropy_flux.cpp
  - solver.py    <-> lib/space/mesh.cpp (computeHybridResidual path),
                     lib/time/hybrid_solver.cpp (RK4 stages)
  - indicator.py <-> lib/time/hybrid_solver.cpp (computeAlpha),
                     lib/time/hybrid_alpha.h (modal_energy_features)
"""

import jax

# The C++ solver runs in double precision; the whole point of this package is
# to reproduce it, so x64 is mandatory (validation vs C++ fails in f32).
jax.config.update("jax_enable_x64", True)

from .basis import GLLBasis
from .solver import Mesh1D, hybrid_residual, rk4_step, time_loop
from .indicator import modal_energy, persson_peraire_alpha, postprocess_alpha
