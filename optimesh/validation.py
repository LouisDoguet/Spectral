"""
Validation plots for residual-based mesh/RBF optimization (see optimize.run_residual).
"""

import numpy as np
import matplotlib.pyplot as plt


def _safe_residual(burgers, val, X, eps_array):
    """Residual at nodes, or NaNs if the RBF system is singular for this config."""
    try:
        return burgers.compute_residual(val, X, eps_array)
    except np.linalg.LinAlgError:
        return np.full_like(np.asarray(val, dtype=float), np.nan)


def plot_convergence(history, ax=None):
    """Plot 1: ||R||^2 vs evaluation, with the best-so-far value highlighted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    objective = history["objective"]
    best_so_far = np.minimum.accumulate(objective)
    evals = np.arange(1, len(objective) + 1)

    ax.plot(evals, objective, '.', color='lightgray', markersize=4, label="evaluations")
    ax.plot(evals, best_so_far, 'r-', linewidth=2, label="best so far")
    ax.set_yscale('log')
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|R\|^2$")
    ax.set_title("Residual Norm Convergence")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    return fig, ax


def plot_node_positions(element_initial, element_optimized, ax=None):
    """Plot 2: node positions before vs after optimization, with the shock location."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.scatter(element_initial.X, np.zeros_like(element_initial.X),
               color='C0', marker='o', alpha=0.5, label='original')
    ax.scatter(element_optimized.X, np.ones_like(element_optimized.X),
               color='C3', marker='x', s=80, label='optimized')
    ax.axvline(element_optimized.shock_pos, color='k', linestyle='--',
               label=f'shock @ {element_optimized.shock_pos:.3f}')

    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Position")
    ax.set_title("Node Clustering: Before vs After")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_solution_profile(element_initial, element_optimized, solution_initial, solution_optimized, burgers, ax=None):
    """Plot 3: RBF interpolant (sum of all RBFs) before/after optimization, against the exact profile."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    fine_x = np.linspace(-1, 1, 500)
    exact = burgers.exact_solution(
        fine_x,
        shock_pos=element_optimized.shock_pos,
        u_left=element_optimized.uL,
        u_right=element_optimized.uR,
    )

    ax.plot(solution_initial.xi, solution_initial.values, '-', color='C0', alpha=0.7, label='initial RBF interpolant')
    ax.plot(solution_optimized.xi, solution_optimized.values, '-', color='C3', linewidth=2, label='optimized RBF interpolant')
    ax.plot(element_initial.X, element_initial.val, 'o', color='C0', alpha=0.5)
    ax.plot(element_optimized.X, element_optimized.val, 's', color='C3', alpha=0.8)
    ax.plot(fine_x, exact, 'k--', label='exact')

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Solution Profile: Before vs After Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_residual_values(element_initial, element_optimized, solution_initial, solution_optimized, burgers, ax=None):
    """Plot 4: |R_i| at each node, before vs after optimization."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    R_initial = _safe_residual(burgers, element_initial.val, element_initial.X, solution_initial.eps)
    R_optimized = _safe_residual(burgers, element_optimized.val, element_optimized.X, solution_optimized.eps)

    idx = np.arange(len(R_initial))
    width = 0.4
    ax.bar(idx - width / 2, np.abs(R_initial), width, color='C0', label='initial')
    ax.bar(idx + width / 2, np.abs(R_optimized), width, color='C3', label='optimized')

    ax.set_yscale('log')
    ax.set_xlabel("Node index")
    ax.set_ylabel(r"$|R_i|$")
    ax.set_title("Residual at Nodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def generate_all_plots(data, show=True, save_dir=None):
    """
    Build the 2x2 validation figure from the dict returned by
    `optimize.run_residual`.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_convergence(data["history"], ax=axes[0, 0])
    plot_node_positions(data["element_initial"], data["element_optimized"], ax=axes[0, 1])
    plot_solution_profile(data["element_initial"], data["element_optimized"],
                           data["solution_initial"], data["solution_optimized"],
                           data["burgers"], ax=axes[1, 0])
    plot_residual_values(data["element_initial"], data["element_optimized"],
                          data["solution_initial"], data["solution_optimized"],
                          data["burgers"], ax=axes[1, 1])

    plt.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/residual_validation.png")
    if show:
        plt.show()

    return fig
