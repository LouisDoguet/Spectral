# Claude Code: Atomic Implementation Tasks

**Goal:** Implement residual-based optimization for Burgers equation in 1-2 days.

**Estimated total time:** 3-4 hours of coding + 1 hour testing/validation.

---

## PHASE 1: CREATE BURGERS CLASS (1.5 hours)

### Task 1.1: Create file `optimesh/burgers.py`
- [ ] Create blank file at `optimesh/burgers.py`
- [ ] Add module docstring explaining Burgers equation
- [ ] Add imports: `numpy as np`, standard library as needed

### Task 1.2: Implement `Burgers.__init__()`
- [ ] Create class `Burgers`
- [ ] Accept parameters: `P=10, shock_intensity=10.0`
- [ ] Store as `self.P`, `self.shock_intensity`
- [ ] Add docstring with parameter descriptions

**Verification:**
```python
b = Burgers(P=10, shock_intensity=10.0)
assert b.P == 10
assert b.shock_intensity == 10.0
```

### Task 1.3: Implement `Burgers.exact_solution()`
- [ ] Accept parameters: `x, shock_pos=0.0, u_left=-1.0, u_right=1.0`
- [ ] Compute tanh profile: `(u_left + u_right)/2 + (u_left - u_right)/2 * np.tanh(self.shock_intensity * (x - shock_pos))`
- [ ] Return array same shape as input `x`

**Verification:**
```python
b = Burgers()
x = np.linspace(-1, 1, 11)
u_exact = b.exact_solution(x, shock_pos=0.0)
assert u_exact.shape == x.shape
assert abs(u_exact[0] - (-1.0)) < 0.01  # Near u_left at x=-1
assert abs(u_exact[-1] - 1.0) < 0.01    # Near u_right at x=1
```

### Task 1.4: Implement `Burgers.flux()`
- [ ] Accept parameter: `u` (1D array)
- [ ] Compute and return: `0.5 * u**2`
- [ ] Maintain input shape

**Verification:**
```python
b = Burgers()
u = np.array([1.0, 2.0, 3.0])
F = b.flux(u)
assert np.allclose(F, [0.5, 2.0, 4.5])
```

### Task 1.5: Implement `Burgers.compute_rbf_derivative_matrix()`
- [ ] Accept parameters: `X` (node positions, shape P+1), `eps_array` (shape parameters, shape P+1)
- [ ] **Part A - Compute collocation matrix A:**
  - For each pair (i, j), compute `r_ij = abs(X[i] - X[j])`
  - `A[i,j] = exp(-(eps_array[j] * r_ij)²)`
  
- [ ] **Part B - Invert A:**
  - `invA = np.linalg.inv(A)`
  - Compute condition number: `cond = np.linalg.cond(A)`
  - Print warning if `cond > 1e10`: `"WARNING: Collocation matrix ill-conditioned (cond={cond:.2e})"`
  - Raise `LinAlgError` if `cond > 1e15`
  
- [ ] **Part C - Compute D_kernel (kernel derivative matrix):**
  - For each pair (i, j):
    - Compute `r_ij = X[i] - X[j]` (SIGNED)
    - If `abs(r_ij) < 1e-14`: set `D_kernel[i,j] = 0`
    - Else:
      - Compute `abs_r = abs(r_ij)`
      - Compute Gaussian derivative: `phi_prime = -2 * (eps_array[j]**2) * abs_r * exp(-(eps_array[j]*abs_r)²)`
      - `D_kernel[i,j] = np.sign(r_ij) * phi_prime`
  
- [ ] **Part D - Compute nodal derivative:**
  - `D = D_kernel @ invA`
  - Return `D`

**Verification:**
```python
b = Burgers()
X = np.linspace(-1, 1, 3)  # 3 nodes
eps_array = 10.0 * np.ones(3)
D = b.compute_rbf_derivative_matrix(X, eps_array)
assert D.shape == (3, 3)
assert not np.any(np.isnan(D))
assert not np.any(np.isinf(D))
```

### Task 1.6: Implement `Burgers.compute_rbf_mass_matrix()`
- [ ] Accept parameters: `X` (shape P+1), `eps_array` (shape P+1), `quad_order=None`
- [ ] **Part A - Set quadrature order:**
  - If `quad_order is None`: `quad_order = max(3 * len(X), 64)`
  
- [ ] **Part B - Get Gauss-Legendre quadrature:**
  - `xi_quad, w_quad = np.polynomial.legendre.leggauss(quad_order)`
  
- [ ] **Part C - Compute A and invA (reuse Task 1.5 logic or call it):**
  - Build collocation matrix A for nodes X, eps_array
  - Invert to get invA
  
- [ ] **Part D - Evaluate RBF basis at quadrature points:**
  - Create `L_rbf` array (shape: (P+1, quad_order))
  - For each quadrature point `q`:
    - For each node `m`, compute `phi_qm = exp(-(eps_array[m] * abs(xi_quad[q] - X[m]))²)`
    - `L_rbf[:,q] = invA @ phi_q` where `phi_q` is the vector of all `phi_qm`
  
- [ ] **Part E - Assemble mass matrix:**
  - `M = L_rbf @ np.diag(w_quad) @ L_rbf.T`
  
- [ ] **Part F - Verify properties:**
  - Symmetry: `symm_error = np.max(np.abs(M - M.T))`
  - If `symm_error > 1e-10`: print `"WARNING: Mass matrix not symmetric (error={symm_error:.2e})"`
  - Eigenvalues: `eigs = np.linalg.eigvalsh(M)`
  - If `min(eigs) < 1e-14`: print `"WARNING: Mass matrix near singular (min eig={min(eigs):.2e})"`
  
- [ ] Return `M`

**Verification:**
```python
b = Burgers()
X = np.linspace(-1, 1, 3)
eps_array = 10.0 * np.ones(3)
M = b.compute_rbf_mass_matrix(X, eps_array)
assert M.shape == (3, 3)
assert np.allclose(M, M.T, atol=1e-9)  # Symmetric
eigs = np.linalg.eigvalsh(M)
assert np.all(eigs > 0)  # Positive definite
```

### Task 1.7: Implement `Burgers.compute_residual()`
- [ ] Accept parameters: `u_values` (shape P+1), `X` (shape P+1), `eps_array` (shape P+1)
- [ ] Try-except wrapper:
  - Inside try:
    1. `F = self.flux(u_values)`
    2. `D = self.compute_rbf_derivative_matrix(X, eps_array)`
    3. `divF = D @ F`
    4. `M = self.compute_rbf_mass_matrix(X, eps_array)`
    5. `R = np.linalg.solve(M, divF)`
  - In except block (if `LinAlgError`): `raise` the error with message
- [ ] Return `R`

**Verification:**
```python
b = Burgers()
X = np.linspace(-1, 1, 5)
u = np.sin(np.pi * X)
eps_array = 10.0 * np.ones(5)
R = b.compute_residual(u, X, eps_array)
assert R.shape == (5,)
assert np.all(np.isfinite(R))
```

### Task 1.8: Implement `Burgers.residual_norm_squared()`
- [ ] Accept parameters: same as `compute_residual()`
- [ ] Try-except:
  - Try: `R = self.compute_residual(...)`
  - Compute: `objective = np.sum(R**2)`
  - Return `objective`
  - Except `LinAlgError`: return `np.inf`
- [ ] Return scalar

**Verification:**
```python
b = Burgers()
X = np.linspace(-1, 1, 5)
u = np.sin(np.pi * X)
eps_array = 10.0 * np.ones(5)
obj = b.residual_norm_squared(u, X, eps_array)
assert isinstance(obj, (float, np.floating))
assert obj >= 0
```

---

## PHASE 2: MODIFY OPTIMIZE.PY (1 hour)

### Task 2.1: Add import
- [ ] At top of `optimize.py`, add: `from burgers import Burgers`

### Task 2.2: Create module-level Burgers instance
- [ ] After imports, add:
  ```python
  # Burgers problem for residual minimization
  burgers_problem = Burgers(P=10, shock_intensity=10.0)
  ```

### Task 2.3: Add new objective function
- [ ] Keep old `interpolation_error()` function (don't delete)
- [ ] Add new function `objective_residual()`:
  ```python
  def objective_residual(params, element, burgers_obj):
      """Minimize ||R||² where R is DG residual."""
      x_s, eps_max = params
      
      try:
          element.cluster(x_s, eps_max)
          S = sol.Solution(element, eps_max)
          objective = burgers_obj.residual_norm_squared(
              u_values=element.val,
              X=element.X,
              eps_array=S.eps
          )
          return objective
      except (np.linalg.LinAlgError, ValueError):
          return np.inf
  ```

### Task 2.4: Create `optimize_residual_case()` function
- [ ] Copy `optimize_case()` and rename to `optimize_residual_case()`
- [ ] Change signature: add `burgers_obj` parameter
- [ ] Inside, change objective function call to use `objective_residual(params, element, burgers_obj)`
- [ ] Keep everything else the same

### Task 2.5: Create `run_residual()` function
- [ ] Copy `run()` function and rename to `run_residual()`
- [ ] Change function body:
  ```python
  def run_residual(P=10, case_seed=None, opt_seed=0, **de_kwargs):
      element = cg.generate_random_case(P=P, seed=case_seed)
      burgers = Burgers(P=P, shock_intensity=10.0)
      result, history = optimize_residual_case(element, burgers, seed=opt_seed, **de_kwargs)
      
      # Store initial state
      element_initial_X = element.X.copy()
      element_initial_val = element.val.copy()
      
      # Apply optimal result
      x_s_opt, eps_opt = result.x
      element.cluster(x_s_opt, eps_opt)
      S = sol.Solution(element, eps_opt)
      
      # Return for validation
      return {
          'element_initial': type('obj', (object,), {'X': element_initial_X, 'val': element_initial_val})(),
          'element_optimized': element,
          'solution': S,
          'result': result,
          'history': history,
          'burgers': burgers
      }
  ```

**Verification:**
```python
from optimize import run_residual
data = run_residual(P=10, case_seed=42)
assert 'history' in data
assert data['history']['objective'][0] >= data['history']['objective'][-1]
```

---

## PHASE 3: CREATE VALIDATION PLOTS (45 minutes)

### Task 3.1: Create file `optimesh/validation.py`
- [ ] Create blank file
- [ ] Add imports: `matplotlib.pyplot as plt, numpy as np`

### Task 3.2: Implement plot_convergence()
- [ ] Function signature: `plot_convergence(history, ax=None)`
- [ ] Extract `objective = history['objective']`
- [ ] Compute `best_so_far = np.minimum.accumulate(objective)`
- [ ] Create plot:
  - Scatter plot of all objectives (gray, small)
  - Line plot of best-so-far (red, thick)
  - Log scale on y-axis
  - Labels: x="iteration", y="||R||²"
  - Title: "Residual Norm Convergence"
- [ ] Return `(fig, ax)`

### Task 3.3: Implement plot_node_positions()
- [ ] Function signature: `plot_node_positions(element_initial, element_optimized, ax=None)`
- [ ] Create scatter plots:
  - Original nodes (blue circles, alpha=0.5)
  - Optimized nodes (red X's, larger)
  - Vertical line at shock position
- [ ] Labels: x="Position", y="(no y-axis meaning)"
- [ ] Title: "Node Clustering: Before vs After"
- [ ] Return `(fig, ax)`

### Task 3.4: Implement plot_solution_profile()
- [ ] Function signature: `plot_solution_profile(element_initial, element_optimized, solution, burgers, ax=None)`
- [ ] Plot three curves:
  - Initial solution at nodes (blue, alpha=0.5, marker='o')
  - Optimized solution at nodes (red, alpha=0.8, marker='s')
  - Exact solution: `burgers.exact_solution(fine_x)` as dashed line (black)
- [ ] Labels: x="x", y="u(x)"
- [ ] Title: "Solution Profile: Before vs After Optimization"
- [ ] Legend
- [ ] Return `(fig, ax)`

### Task 3.5: Implement plot_residual_values()
- [ ] Function signature: `plot_residual_values(element_initial, element_optimized, burgers, ax=None)`
- [ ] Compute residuals at initial and optimized states
- [ ] Create bar chart:
  - Initial residual: blue bars
  - Optimized residual: red bars
  - Grouped by node index
- [ ] Y-axis log scale
- [ ] Labels: x="Node index", y="|R_i|"
- [ ] Title: "Residual at Nodes"
- [ ] Return `(fig, ax)`

### Task 3.6: Implement generate_all_plots()
- [ ] Function signature: `generate_all_plots(data, show=True, save_dir=None)`
- [ ] Create 2×2 subplot grid
- [ ] Call each plot function, filling in the 4 axes
- [ ] Tight layout
- [ ] If `save_dir`: save to PNG: `plt.savefig(f"{save_dir}/residual_validation.png")`
- [ ] If `show`: `plt.show()`
- [ ] Return figure

---

## PHASE 4: TEST & VALIDATE (1 hour)

### Task 4.1: Test imports
- [ ] Run in terminal:
  ```bash
  cd optimesh
  python -c "from burgers import Burgers; from optimize import optimize_residual_case; print('OK')"
  ```
- [ ] Should print `OK` with no errors

### Task 4.2: Test Burgers class
- [ ] Run in Python/notebook:
  ```python
  from burgers import Burgers
  import numpy as np
  
  b = Burgers(P=10)
  X = np.linspace(-1, 1, 11)
  eps = 10 * np.ones(11)
  u = np.sin(np.pi * X)
  
  # Test each method
  F = b.flux(u)
  D = b.compute_rbf_derivative_matrix(X, eps)
  M = b.compute_rbf_mass_matrix(X, eps)
  R = b.compute_residual(u, X, eps)
  obj = b.residual_norm_squared(u, X, eps)
  
  print(f"Flux: {F.shape}, min={F.min():.2e}, max={F.max():.2e}")
  print(f"D: {D.shape}, cond={np.linalg.cond(D):.2e}")
  print(f"M: {M.shape}, det={np.linalg.det(M):.2e}")
  print(f"R: {R.shape}, norm={np.linalg.norm(R):.2e}")
  print(f"Objective: {obj:.2e}")
  ```
- [ ] All should print without error

### Task 4.3: Run full optimization
- [ ] In Python:
  ```python
  from optimize import run_residual
  data = run_residual(P=10, case_seed=42, opt_seed=0)
  ```
- [ ] Should complete in < 30 seconds
- [ ] Check `data['history']['objective']` is decreasing

### Task 4.4: Generate validation plots
- [ ] In Python:
  ```python
  from validation import generate_all_plots
  generate_all_plots(data, show=True, save_dir=None)
  ```
- [ ] Should show 4 subplots without error
- [ ] Visually inspect:
  - Convergence plot: ||R||² decreasing monotonically?
  - Node plot: nodes cluster around shock?
  - Solution plot: optimized smoother than original?
  - Residual plot: values smaller after optimization?

### Task 4.5: Verify correctness
- [ ] Check condition numbers:
  ```python
  print(f"Min cond(D): {min_cond_D}")
  print(f"Min cond(M): {min_cond_M}")
  ```
  - Should be < 1e10
  
- [ ] Check objective decrease:
  ```python
  obj_initial = data['history']['objective'][0]
  obj_final = data['history']['objective'][-1]
  print(f"Initial: {obj_initial:.2e}, Final: {obj_final:.2e}, Ratio: {obj_initial/obj_final:.1f}x")
  ```
  - Should decrease by at least 2x

---

## PHASE 5: DOCUMENTATION & CLEANUP (15 minutes)

### Task 5.1: Add docstrings
- [ ] Every class and method has docstring explaining purpose, inputs, returns
- [ ] Example docstring format:
  ```python
  def compute_residual(self, u_values, X, eps_array):
      """
      Compute the DG weak-form residual R = M⁻¹ @ (D @ F).
      
      Parameters
      ----------
      u_values : array_like, shape (P+1,)
          Solution values at nodes.
      X : array_like, shape (P+1,)
          Node positions in [-1, 1].
      eps_array : array_like, shape (P+1,)
          RBF shape parameters per node.
      
      Returns
      -------
      R : ndarray, shape (P+1,)
          DG residual at nodes.
      
      Raises
      ------
      LinAlgError
          If mass matrix is singular.
      """
  ```

### Task 5.2: Code comments
- [ ] Add inline comments for complex sections (e.g., derivative matrix computation)
- [ ] Keep comments brief and focused on "why", not "what"

### Task 5.3: Test file cleanup
- [ ] Remove any test code from main modules
- [ ] Move testing logic to separate test blocks or separate test file

### Task 5.4: Final verification
- [ ] Run through entire pipeline once more:
  ```bash
  cd optimesh
  python -c "from optimize import run_residual; from validation import generate_all_plots; data = run_residual(P=10, case_seed=42); generate_all_plots(data, show=False)"
  ```
- [ ] Should complete without errors or warnings

---

## Summary Checklist

- [ ] All 8 Burgers methods implemented and tested
- [ ] optimize.py modified with new objective and run function
- [ ] validation.py created with 4 plot functions
- [ ] Full pipeline runs: `run_residual()` → `generate_all_plots()`
- [ ] Convergence plot shows ||R||² decreasing
- [ ] Solution visually improves after optimization
- [ ] All docstrings complete
- [ ] No import errors or undefined variables

---

## Expected Timeline

| Phase | Time | Status |
|-------|------|--------|
| Phase 1 (Burgers class) | 1.5 hours | - |
| Phase 2 (Modify optimize.py) | 1 hour | - |
| Phase 3 (Validation plots) | 45 min | - |
| Phase 4 (Test & validate) | 1 hour | - |
| Phase 5 (Documentation) | 15 min | - |
| **TOTAL** | **4-4.5 hours** | - |

**If you complete Phase 4 by EOD tomorrow, you have validated the approach and can begin C++ implementation next week.**

