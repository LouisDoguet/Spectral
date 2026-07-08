import muscl.case as case
import muscl.solver as solver

mesh = case.NN_Mesh(n_batch=3)
mesh.setInitialCondition(rand_seed=2)

f = lambda u: 0.5*u**2
s = lambda u: u

mesh.plot()
sol = solver.NN_MUSCL_Solver(mesh, f, s, 10e-5)
sol.solve(0.2)
mesh.plot()