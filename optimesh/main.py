from optimize import run_residual
from validation import generate_all_plots

data = run_residual(P=10, opt_seed=0)
generate_all_plots(data, show=True)
