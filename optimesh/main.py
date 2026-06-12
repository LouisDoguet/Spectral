import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import element as el
import solution as sol

N=1000
xi = np.linspace(-1,1,N)

E = el.Element(P=10, uL=4, uR=0, shock_pos=0.2, shock_intensity=100)
E.cluster(x0=0.2, epsilon=1)
S = sol.Solution(element=E, eps=10)
E.display(S)
S.plot_rbfs()