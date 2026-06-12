import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import case_generator as cg
import solution as sol

# Search bounds for (x0, eps_cluster, eps_solution)
BOUNDS = [(-1.0, 1.0), (1e-3, 100.0), (1e-2, 200.0)]
PARAM_NAMES = ["$x_0$", r"$\epsilon_{cluster}$", r"$\epsilon_{solution}$"]


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


if __name__ == "__main__":
    run()
