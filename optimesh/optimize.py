from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import case_generator as cg
import solution as sol
from burgers import Burgers

# Search bounds for (x0, eps_cluster, eps_solution)
BOUNDS = [(-1.0, 1.0), (1e-3, 100.0), (1e-2, 200.0)]
PARAM_NAMES = ["$x_0$", r"$\epsilon_{cluster}$", r"$\epsilon_{solution}$"]

# Search bounds for residual minimization: (x_s, eps_max)
BOUNDS_RESIDUAL = [(-1.0, 1.0), (1e-3, 100.0)]
PARAM_NAMES_RESIDUAL = ["$x_s$", r"$\epsilon_{max}$"]


def interpolation_error(params, element):
    """RMS error between the RBF interpolant and the exact profile."""
    rbf = None
    x0, eps_cluster, eps_solution = params
    element.cluster(x0, eps_cluster)
    #rbf = lambda r, eps: 1 / np.sqrt(1 + eps*r**2)
    try:
        if rbf:
            S = sol.Solution(element, eps_solution, rbf)
        else:
            S = sol.Solution(element, eps_solution)
        exact = element.discontinuity(S.xi)
        err = np.sqrt(np.mean((S.values - exact) ** 2))
    except np.linalg.LinAlgError:
        err = np.inf
    return err if np.isfinite(err) else np.inf


def optimize_case(element, bounds=BOUNDS, seed=0, **de_kwargs):
    """
    Searches for the (x0, eps_cluster, eps_solution) that minimize the RBF
    interpolation error on `element`. Returns the scipy result together with
    a record of every evaluated point, for plotting the optimization process.
    """
    de_kwargs.setdefault("maxiter", 10000)
    history = {"params": [], "error": []}

    def objective(params):
        err = interpolation_error(params, element)
        history["params"].append(np.array(params, dtype=float))
        history["error"].append(err)
        return err

    result = opt.differential_evolution(objective, bounds, seed=seed, polish=True, **de_kwargs)

    history["params"] = np.array(history["params"])
    history["error"] = np.array(history["error"])
    return result, history


def objective_residual(params, element, burgers_obj, cond_penalty_weight=50.0, cond_log10_threshold=8.0):
    """
    Objective: ||R(u)||^2 + a penalty for an ill-conditioned collocation matrix.

    The penalty is `cond_penalty_weight * max(0, log10(cond(A)) - cond_log10_threshold)`,
    i.e. zero while cond(A) stays below 10**cond_log10_threshold and grows linearly
    in log10(cond(A)) beyond that. This discourages configurations that achieve a
    small nodal residual only because A is nearly singular (see the RBF interpolant
    overshoot/ringing observed between nodes for such configurations).
    """
    x_s, eps_max = params

    try:
        element.cluster(x_s, eps_max)
        S = sol.Solution(element, eps_max)

        R2 = burgers_obj.residual_norm_squared(
            u_values=element.val,
            X=element.X,
            eps_array=S.eps,
        )
        if not np.isfinite(R2):
            return np.inf

        cond = burgers_obj.collocation_condition_number(element.X, S.eps)
        penalty = cond_penalty_weight * max(0.0, np.log10(cond) - cond_log10_threshold)

        return R2 + penalty
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def optimize_residual_case(element, burgers_obj, bounds=BOUNDS_RESIDUAL, seed=0,
                            cond_penalty_weight=50.0, cond_log10_threshold=8.0, **de_kwargs):
    """
    Searches for the (x_s, eps_max) that minimize the DG residual norm
    ||R(u)||^2 (plus a conditioning penalty, see `objective_residual`) on
    `element`. Returns the scipy result together with a record of every
    evaluated point, for plotting the optimization process.
    """
    # The objective is `inf` for ill-conditioned configurations, which keeps
    # differential_evolution's convergence check from triggering early
    # (it then runs to `maxiter`). 300 generations is ample for this
    # 2-parameter search, so keep the default modest.
    de_kwargs.setdefault("maxiter", 300)
    history = {"params": [], "objective": []}

    def objective(params):
        obj = objective_residual(params, element, burgers_obj,
                                  cond_penalty_weight=cond_penalty_weight,
                                  cond_log10_threshold=cond_log10_threshold)
        history["params"].append(np.array(params, dtype=float))
        history["objective"].append(obj)
        return obj

    result = opt.differential_evolution(objective, bounds, seed=seed, polish=True, **de_kwargs)

    history["params"] = np.array(history["params"])
    history["objective"] = np.array(history["objective"])
    return result, history


def _running_best(history):
    """For each evaluation, the (error, params) of the best point found so far."""
    errors = history["error"]
    best_idx = np.empty(len(errors), dtype=int)
    current = 0
    for i, e in enumerate(errors):
        if e < errors[current]:
            current = i
        best_idx[i] = current
    return errors[best_idx], history["params"][best_idx]


def plot_convergence(history, ax=None, disp=True):
    """Error of every evaluated point, with the best-so-far value highlighted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    best_error, _ = _running_best(history)
    evals = np.arange(1, len(history["error"]) + 1)

    ax.plot(evals, history["error"], '.', color='lightgray', markersize=4, label="evaluations")
    ax.plot(evals, best_error, 'r-', linewidth=2, label="best so far")
    ax.set_yscale('log')
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("RMS error")
    ax.set_title("Optimization convergence")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    if disp:
        plt.show()
    return fig, ax


def plot_parameter_history(history, disp=True):
    """Best-so-far value of each search parameter across evaluations."""
    _, best_params = _running_best(history)
    evals = np.arange(1, len(history["error"]) + 1)

    fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(6, 7), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        ax.plot(evals, history["params"][:, k], '.', color='lightgray', markersize=4)
        ax.plot(evals, best_params[:, k], 'r-', linewidth=2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("function evaluation")

    if disp:
        plt.show()
    return fig, axes


def plot_result(element, result, disp=True):
    """Apply the optimized parameters to `element` and display the resulting solution."""
    x0, eps_cluster, eps_solution = result.x
    element.cluster(x0, eps_cluster)
    S = sol.Solution(element, eps_solution)

    print(f"Optimal x0={x0:.4f}, eps_cluster={eps_cluster:.4f}, eps_solution={eps_solution:.4f}")
    print(f"RMS error: {result.fun:.3e}")

    element.display(S)
    S.plot_rbfs(disp=disp)
    return S


def run(P=10, case_seed=None, opt_seed=0, **de_kwargs):
    """
    Generates a random shock case and finds the (x0, eps_cluster, eps_solution)
    that minimize the RBF interpolation error, plotting the optimization
    process and the resulting solution.
    """
    element = cg.generate_random_case(P=P, seed=case_seed)
    result, history = optimize_case(element, seed=opt_seed, **de_kwargs)

    plot_result(element, result)
    plot_convergence(history)
    plot_parameter_history(history)

    return element, result, history


def run_residual(P=10, case_seed=None, opt_seed=0, **de_kwargs):
    """
    Generates a random shock case and finds the (x_s, eps_max) that minimize
    the DG residual norm ||R(u)||^2. Returns a dict with the initial and
    optimized element states, ready for `validation.generate_all_plots`.
    """
    element = cg.generate_random_case(P=P, seed=case_seed)
    burgers = Burgers(P=P, shock_intensity=10.0)

    # Initial grid is uniform, so local_spacing() is h0 = 2/P everywhere.
    element_initial = SimpleNamespace(
        X=element.X.copy(),
        val=element.val.copy(),
        P=element.P,
        local_spacing=lambda: np.full(element.P + 1, 2.0 / element.P),
    )

    result, history = optimize_residual_case(element, burgers, seed=opt_seed, **de_kwargs)

    x_s_opt, eps_opt = result.x

    # RBF interpolant on the pre-optimization (uniform) grid, for comparison.
    solution_initial = sol.Solution(element_initial, eps_opt)

    element.cluster(x_s_opt, eps_opt)
    solution_optimized = sol.Solution(element, eps_opt)

    return {
        "element_initial": element_initial,
        "element_optimized": element,
        "solution_initial": solution_initial,
        "solution_optimized": solution_optimized,
        "result": result,
        "history": history,
        "burgers": burgers,
    }


if __name__ == "__main__":
    run()
