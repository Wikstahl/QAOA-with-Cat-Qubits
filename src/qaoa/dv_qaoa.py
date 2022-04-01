import pickle
import numpy as np
import networkx as nx
from scipy.optimize import brute, minimize, Bounds, shgo, differential_evolution
from cvqaoa.circ import Circ


def interpolation(x0):
    """
    INTERPOLATION-BASED STRATEGY
    Description
    -----------
    Uses linear interpolation to produce a good starting point
    for optimizing QAOA as one iteratively increases the
    level p.
    Parameters
    ----------
    opt : 1-2(p-1) array (row vector)
         Optimal angels for level p-1
    Returns
    -------
    x0 : 2-p array (row vector)
         Starting-points for level p.
    """
    # declare variabels
    p = int(len(x0) / 2)
    gamma_opt = x0[:p]
    beta_opt = x0[p:]
    gamma0 = []
    beta0 = []

    # gamma0 and beta0
    for j in range(1, p + 2):
        if j == 1:
            gamma0.append((p - j + 1) / p * gamma_opt[j - 1])
            beta0.append((p - j + 1) / p * beta_opt[j - 1])
        elif j == (p + 1):
            gamma0.append((j - 1) / p * gamma_opt[j - 2])
            beta0.append((j - 1) / p * beta_opt[j - 2])
        else:
            gamma0.append(
                (j - 1) / p * gamma_opt[j - 2] + (p - j + 1) / p * gamma_opt[j - 1])
            beta0.append(
                (j - 1) / p * beta_opt[j - 2] + (p - j + 1) / p * beta_opt[j - 1])
    return np.array([gamma0, beta0]).flatten()


# pick a level p that you want to optimize
level = 1

# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/instances/max_cut_" + str(idx) + "/"
    # Load graph from path
    with open(path + "graph", 'rb') as pickle_file:
        graph = pickle.load(pickle_file)
    # Create object
    circ = Circ(graph)
    if level == 1:
        # Optimization bounds
        ranges = ((0, np.pi), (0, np.pi / 2))
        # Brute force on 100 x 100 grid
        res = brute(circ.optimize_brute_qaoa, ranges, args=(["DV"]), Ns=100,
                    full_output=True, finish=None, workers=-1)
    if level > 1:
        ranges = ((0, np.pi), (0, np.pi / 2))
        # load data
        res_dv = pickle.load(
            open(path + f"qaoa_parameters_dv_level_{level-1}", "rb"))
        # use interpolation method
        #x0 = interpolation(res_dv[0])
        # define bounds
        bounds_gamma = ((0, np.pi),) * level
        bounds_beta = ((0, np.pi / 2),) * level
        bounds = bounds_gamma + bounds_beta
        # find the optimal angles
        res = differential_evolution(circ.optimize_qaoa, bounds, args=(["DV"]))

    # Save results
    filename = path + f"qaoa_parameters_dv_level_{level}"
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
