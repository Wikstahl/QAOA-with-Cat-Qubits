import pickle
import numpy as np
from os.path import exists
from tqdm import tqdm
import networkx as nx
from scipy.optimize import brute, minimize, Bounds, shgo, differential_evolution
from scipy import optimize
import multiprocessing
from qaoa_with_cat_qubits.circ import Circ


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

num_qubits = 10

# pick a level p that you want to optimize
for level in range(1,6+1):
    # Loop over all instances
    for idx in tqdm(range(30)):
        # Path
        path = f"../../data/instances/max_cut_{idx}_num_qubits_{num_qubits}/"
        # Load graph from path
        with open(path + "graph", 'rb') as pickle_file:
            graph = pickle.load(pickle_file)
        # Create object
        circ = Circ(graph)
        fun = circ.optimize_qaoa # optimization function
        # Use brute force optimization for level 1
        if level == 1:
            # Optimization bounds
            ranges = ((0, np.pi), (0, np.pi / 2))
            # Brute force on 100 x 100 grid
            res = brute(circ.optimize_brute_qaoa, ranges, args=(["NoiseFree"]), Ns=100,
                        full_output=True, finish=None, workers=-1)
        # Use Multistart for level > 1
        if level > 1:
            # lower and upper bounds
            bounds_gamma = ((0, np.pi),) * level
            bounds_beta = ((0, np.pi / 2),) * level
            bounds = bounds_gamma + bounds_beta

            # Load level - 1 optimal angles
            with open(path +  f"qaoa_parameters_level_{level-1}", 'rb') as pickle_file:
                prev_res = pickle.load(pickle_file)
            xGuess = interpolation(prev_res.x)
           
            options = {'disp': None, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-06, 'eps': 1e-05,
                    'maxfun': 500, 'maxiter': 500, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
            res = optimize.minimize(fun, xGuess, args=(
                ["NoiseFree"]), bounds=bounds, method="L-BFGS-B", options=options)

        # Save results
        filename = f"../../data/instances/max_cut_{idx}_num_qubits_{num_qubits}/qaoa_parameters_{level}"

        with open(filename, 'wb') as f:
            pickle.dump(res, f)

