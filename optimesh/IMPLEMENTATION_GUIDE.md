# Implementation Guide: Residual-Based Optimization for Burgers Equation

## Overview

This guide takes you from current L₂ minimization to **residual-based minimization** in ~1-2 days of work. The focus is on **exact, optimized implementation** in Python that validates your technical approach before C++ porting.

**Key changes:**
- New `Burgers` class encapsulates the PDE and residual computation
- Residual = M⁻¹ @ (D @ F) computed exactly (not approximated)
- Unconstrained optimization on ||R||²
- Full validation plots showing before/after improvement

---

## Part 1: Create `burgers.py` (New File)

This file contains the entire Burgers problem setup and residual computation logic.

### File Location
```
optimesh/burgers.py
```

### Class Structure

```python
class Burgers:
    """
    Burgers equation: ∂u/∂t + ∂(u²/2)/∂x = 0
    
    Responsible for:
    - Defining the exact solution (Sod problem variant)
    - Computing flux F(u) = u²/2
    - Computing RBF derivative and mass matrices
    - Computing DG residual R = M⁻¹ @ (D @ F)
    """
```

### Key Methods to Implement

#### 1. `__init__(self, P=10, shock_intensity=10.0)`
**Purpose:** Set up the Burgers problem with specific shock profile.

**Input parameters:**
- `P`: polynomial order (number of nodes = P+1)
- `shock_intensity`: steepness of tanh profile (controls smoothness of transition)

**Store:**
- Reference domain parameters (always [-1, 1])
- Shock intensity for exact solution generation
- Pre-allocate any matrices/arrays you'll reuse

**Design choice:** Keep this **independent of Element**—Burgers manages the physics, Element manages nodes.

---

#### 2. `exact_solution(self, x)`
**Purpose:** Return u_exact at arbitrary points x for error metrics (optional for this phase).

**Input:** 
- `x`: array of positions in [-1, 1]

**Return:** 
- `u_exact`: array of solution values

**Formula:** Use a tanh profile:
```
u_exact(x) = (u_left + u_right)/2 + (u_left - u_right)/2 * tanh(shock_intensity * (x - shock_pos))
```

For Sod variant: `u_left = -1.0, u_right = 1.0, shock_pos = 0.0` (adjustable).

**Note:** This is for validation only; not used in residual computation.

---

#### 3. `flux(self, u)`
**Purpose:** Compute Burgers flux F(u) = u²/2.

**Input:** 
- `u`: solution values at nodes (shape: (P+1,))

**Return:** 
- `F`: flux values (shape: (P+1,))

**Implementation:** 
```
F = 0.5 * u**2
```

**Design choice:** Hardcoded in Burgers but structured so you can add other fluxes later (e.g., `if self.flux_type == 'burgers': ...`).

---

#### 4. `compute_rbf_derivative_matrix(self, X, eps_array)`
**Purpose:** Compute the RBF derivative matrix D for the current nodal configuration.

**Inputs:**
- `X`: node positions, shape (P+1,)
- `eps_array`: shape parameters per node, shape (P+1,)

**Returns:**
- `D`: derivative matrix, shape (P+1, P+1)

**Algorithm:**

1. **Compute A (collocation matrix):**
   ```
   For i, j in 0..P:
       r_ij = |X[i] - X[j]|
       A[i,j] = exp(-(eps_array[j] * r_ij)²)
   ```

2. **Invert A to get A⁻¹:**
   ```
   invA = np.linalg.inv(A)
   
   # Check condition number
   cond = np.linalg.cond(A)
   if cond > 1e10:
       print(f"WARNING: Collocation matrix ill-conditioned (cond={cond:.2e})")
   if cond > 1e15:
       raise np.linalg.LinAlgError("Collocation matrix is singular")
   ```

3. **Compute D_kernel (kernel derivative matrix):**
   ```
   For i, j in 0..P:
       r_ij = X[i] - X[j]  # SIGNED difference
       if |r_ij| < 1e-14:
           D_kernel[i,j] = 0  # On-diagonal
       else:
           abs_r = |r_ij|
           # ∂φ/∂r for Gaussian: -2ε² r exp(-(εr)²)
           phi_prime = -2 * (eps_array[j]**2) * abs_r * exp(-(eps_array[j]*abs_r)²)
           D_kernel[i,j] = sign(r_ij) * phi_prime
   ```

4. **Compute D_nodal:**
   ```
   D = D_kernel @ invA
   ```

**Design choice:** Store `invA` during this call so it can be reused for mass matrix (if efficiency matters later).

**Return:** `D` (do not return invA separately; you'll recompute it if needed).

---

#### 5. `compute_rbf_mass_matrix(self, X, eps_array, quad_order=None)`
**Purpose:** Compute the RBF mass matrix M using Gauss-Legendre quadrature.

**Inputs:**
- `X`: node positions, shape (P+1,)
- `eps_array`: shape parameters, shape (P+1,)
- `quad_order`: number of quadrature points (default: 3*(P+1))

**Returns:**
- `M`: mass matrix, shape (P+1, P+1)

**Algorithm:**

1. **Get Gauss-Legendre quadrature:**
   ```
   if quad_order is None:
       quad_order = max(3 * len(X), 64)
   
   xi_quad, w_quad = np.polynomial.legendre.leggauss(quad_order)
   ```

2. **Compute collocation matrix A and its inverse:**
   ```
   A, invA = [same as in compute_rbf_derivative_matrix]
   ```

3. **Evaluate RBF basis at quadrature points:**
   ```
   L_rbf = np.zeros((P+1, quad_order))
   
   For q in 0..quad_order:
       For m in 0..P:
           r_qm = |xi_quad[q] - X[m]|
           phi_qm = exp(-(eps_array[m] * r_qm)²)
       
       L_rbf[:,q] = invA @ phi_q  # phi_q = [phi_qm for all m]
   ```

4. **Assemble mass matrix:**
   ```
   M = L_rbf @ np.diag(w_quad) @ L_rbf.T
   ```

5. **Verify symmetric positive definite:**
   ```
   # Check symmetry
   symm_error = np.max(np.abs(M - M.T))
   if symm_error > 1e-10:
       print(f"WARNING: Mass matrix not symmetric (error={symm_error:.2e})")
   
   # Check eigenvalues
   eigs = np.linalg.eigvalsh(M)
   if np.min(eigs) < 1e-14:
       print(f"WARNING: Mass matrix near singular (min eigenvalue={np.min(eigs):.2e})")
   ```

**Return:** `M`

**Design choice:** Compute both A and invA fresh each time (no caching for now; premature optimization).

---

#### 6. `compute_residual(self, u_values, X, eps_array)`
**Purpose:** Compute the DG weak-form residual R = M⁻¹ @ (D @ F).

**Inputs:**
- `u_values`: solution at nodes, shape (P+1,)
- `X`: node positions, shape (P+1,)
- `eps_array`: shape parameters, shape (P+1,)

**Returns:**
- `R`: residual at nodes, shape (P+1,)

**Algorithm:**

1. **Compute flux:**
   ```
   F = self.flux(u_values)
   ```

2. **Compute derivative matrix:**
   ```
   D = self.compute_rbf_derivative_matrix(X, eps_array)
   ```

3. **Compute divergence:**
   ```
   divF = D @ F
   ```

4. **Compute mass matrix:**
   ```
   M = self.compute_rbf_mass_matrix(X, eps_array)
   ```

5. **Solve M @ R = divF:**
   ```
   try:
       R = np.linalg.solve(M, divF)
   except np.linalg.LinAlgError as e:
       raise np.linalg.LinAlgError(f"Mass matrix inversion failed: {e}")
   ```

**Return:** `R`

**Design choice:** Call this from optimize.py; do NOT store intermediate matrices (compute fresh each iteration).

---

#### 7. `residual_norm_squared(self, u_values, X, eps_array)`
**Purpose:** Wrapper for the optimization objective: ||R||².

**Inputs:**
- Same as `compute_residual()`

**Returns:**
- `objective`: scalar ||R||² = ∑ R_i²

**Implementation:**
```python
def residual_norm_squared(self, u_values, X, eps_array):
    try:
        R = self.compute_residual(u_values, X, eps_array)
        return np.sum(R**2)
    except np.linalg.LinAlgError:
        return np.inf
```

**Design choice:** Catch singular matrix errors and return `np.inf` (tells optimizer this config is infeasible).

---

### Summary: What `burgers.py` Contains

```python
import numpy as np

class Burgers:
    def __init__(self, P=10, shock_intensity=10.0):
        # Store parameters
        
    def exact_solution(self, x, shock_pos=0.0, u_left=-1.0, u_right=1.0):
        # Return tanh-based exact solution
        
    def flux(self, u):
        # Return u²/2
        
    def compute_rbf_derivative_matrix(self, X, eps_array):
        # Return D matrix (exact from Gaussian derivatives)
        
    def compute_rbf_mass_matrix(self, X, eps_array, quad_order=None):
        # Return M matrix (via Gauss-Legendre quadrature)
        
    def compute_residual(self, u_values, X, eps_array):
        # Return R = M⁻¹ @ (D @ F(u))
        
    def residual_norm_squared(self, u_values, X, eps_array):
        # Return ||R||²
```

---

## Part 2: Modify `solution.py`

The Solution class already does RBF interpolation. **Minimal change needed.**

### What to Add/Change

1. **Store A and invA (optional, for future efficiency):**
   ```python
   class Solution:
       def __init__(self, element, eps):
           # ... existing code ...
           self.A = A  # Store collocation matrix
           self.invA = invA  # Store its inverse
   ```

   This lets you skip recomputation if needed, but **not required for now.**

2. **No other changes required** — Solution is independent of Burgers.

---

## Part 3: Modify `optimize.py`

This is where the residual minimization happens.

### Changes Required

#### 1. Import Burgers
```python
from burgers import Burgers
```

#### 2. Create a Burgers instance at module level
```python
burgers_problem = Burgers(P=10, shock_intensity=10.0)
```

#### 3. Replace the objective function
**Old:**
```python
def interpolation_error(params, element):
    x_s, eps_cluster, eps_solution = params
    element.cluster(x_s, eps_cluster)
    S = sol.Solution(element, eps_solution)
    exact = element.discontinuity(S.xi)
    err = np.sqrt(np.mean((S.values - exact)**2))
    return err if np.isfinite(err) else np.inf
```

**New:**
```python
def objective_residual(params, element, burgers_obj, flux_type='burgers'):
    """
    Minimize ||R||² where R is the DG residual.
    
    Args:
        params: (x_s, eps_max)
        element: Element object
        burgers_obj: Burgers problem instance
        flux_type: currently only 'burgers' (hardcoded)
    
    Returns:
        objective: ||R||² (scalar)
    """
    x_s, eps_max = params
    
    try:
        # 1. Cluster nodes around shock
        element.cluster(x_s, eps_max)
        
        # 2. Compute RBF solution at nodes
        S = sol.Solution(element, eps_max)
        
        # 3. Compute residual norm
        objective = burgers_obj.residual_norm_squared(
            u_values=element.val,
            X=element.X,
            eps_array=S.eps  # Adaptive per-node parameters
        )
        
        return objective
    
    except (np.linalg.LinAlgError, ValueError):
        return np.inf
```

#### 4. Update optimize_case() signature
```python
def optimize_case(element, burgers_obj, bounds=BOUNDS, seed=0, **de_kwargs):
    """
    Search for optimal (x_s, eps_max) minimizing ||R||².
    """
    de_kwargs.setdefault("maxiter", 10000)
    history = {"params": [], "objective": []}

    def objective_tracked(params):
        obj = objective_residual(params, element, burgers_obj)
        history["params"].append(np.array(params))
        history["objective"].append(obj)
        return obj

    result = opt.differential_evolution(
        objective_tracked, 
        bounds, 
        seed=seed, 
        polish=True, 
        **de_kwargs
    )

    history["params"] = np.array(history["params"])
    history["objective"] = np.array(history["objective"])
    
    return result, history
```

#### 5. Update run() function
```python
def run(P=10, case_seed=None, opt_seed=0, **de_kwargs):
    """
    Generate test case and minimize residual.
    """
    # Generate test element (Sod-like shock)
    element = cg.generate_random_case(P=P, seed=case_seed)
    
    # Create Burgers problem
    burgers = Burgers(P=P, shock_intensity=10.0)
    
    # Run optimization
    result, history = optimize_case(element, burgers, seed=opt_seed, **de_kwargs)

    # Visualize results
    plot_result_residual(element, burgers, result)
    plot_convergence(history, title="Residual Norm Convergence")
    plot_parameter_history(history)

    return element, result, history, burgers
```

---

## Part 4: Create `validation.py` (New File)

Generates the four validation plots.

### File Location
```
optimesh/validation.py
```

### What It Does

```python
import matplotlib.pyplot as plt
import numpy as np

class ResidualValidator:
    """Generate validation plots for residual-based optimization."""
    
    def __init__(self, element, burgers, result):
        self.element = element
        self.burgers = burgers
        self.result = result
    
    def plot_convergence(self):
        """Plot 1: ||R||² vs iteration"""
        # Extract convergence data from history
        # Plot on log scale
        
    def plot_node_positions(self):
        """Plot 2: Before/after node clustering"""
        # Show original equidistant nodes vs optimized
        
    def plot_solution_profile(self):
        """Plot 3: Solution before/after with exact"""
        # RBF interpolant at original vs optimized configs
        # Compare to exact solution if available
        
    def plot_residual_values(self):
        """Plot 4: Residual at nodes"""
        # Bar plot of |R_i| before/after
        
    def generate_all(self, show=True, save_dir=None):
        """Generate all four plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # ... populate axes ...
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/residual_validation.png")
        if show:
            plt.show()
```

---

## Part 5: Integration Checklist

Before you declare "done," verify:

### Code Structure
- [ ] `burgers.py` created with all 7 methods
- [ ] `optimize.py` imports `Burgers` and calls new objective function
- [ ] `validation.py` created with 4 plot functions
- [ ] No import errors: `python -c "from optimesh import burgers, optimize"`

### Correctness Checks
- [ ] `burgers.flux(np.array([1, 2, 3]))` returns `[0.5, 2.0, 4.5]` ✓
- [ ] `compute_rbf_derivative_matrix()` returns (P+1) × (P+1) matrix ✓
- [ ] `compute_rbf_mass_matrix()` returns symmetric positive definite matrix ✓
- [ ] `compute_residual()` returns (P+1,) array without NaN ✓
- [ ] `residual_norm_squared()` returns scalar in [0, ∞) ✓

### Conditioning Checks
- [ ] Collocation matrix condition number < 1e10 for typical ε_max values
- [ ] Mass matrix eigenvalues all > 1e-12
- [ ] Optimizer doesn't crash on singular matrices (returns inf)

### Convergence Checks
- [ ] Run optimization on Sod case
- [ ] Objective ||R||² decreases monotonically over iterations ✓
- [ ] Final ||R||² < initial by at least 50%
- [ ] Optimized nodes visually cluster around shock

### Plot Quality Checks
- [ ] 4 plots generate without error
- [ ] Convergence plot shows clear decrease
- [ ] Solution plot shows smoother profile after optimization
- [ ] Residual plot shows values reduced

---

## Key Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **D matrix** | Exact Gaussian formula | Accuracy; no approximation |
| **M matrix** | Gauss-Legendre quadrature | 3×(P+1) points; balanced accuracy/speed |
| **Constraint** | Unconstrained (for now) | Simplicity; add if residuals remain |
| **Flux** | Hardcoded in Burgers | Burgers for proof-of-concept; extensible design |
| **Error handling** | Raise on singular, warn on ill-conditioning | Fail-fast for debugging |
| **Recomputation** | Fresh each iteration | Correctness over speed in validation phase |

---

## Expected Results

After running the optimization on a Sod-type problem:

1. **Convergence plot:** Smooth monotonic decrease in ||R||² from ~10 to ~0.1–1.0
2. **Node positions:** Nodes cluster tightly around shock (x_s ≈ 0.0), leave equidistant elsewhere
3. **Solution profile:** Optimized solution smoother, fewer wiggles near shock
4. **Residual values:** Max |R_i| at shocked nodes decreases significantly

If you see these, **the approach is validated** and ready for C++ implementation.

---

## Notes for C++ Port

When you move this to C++:

1. **D matrix:** Use your existing RBF derivative computation
2. **M matrix:** Implement Gauss-Legendre integration (or lookup table quadrature rules)
3. **Residual:** Reuse Element::computeDivFlux() + applyMassInverse()
4. **Optimizer:** Integrate into your time-stepping loop; recompute R on-the-fly per element

The Python version is a **complete, standalone validator.** C++ will be the production implementation.

