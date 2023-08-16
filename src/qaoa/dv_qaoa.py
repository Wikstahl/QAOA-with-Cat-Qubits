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


# pick a level p that you want to optimize
alpha = 1.36
cutoff = 20
num_qubits = 8

for level in range(1,6):
    # Loop over all instances
    for idx in tqdm(range(30)):
        print(idx)
        # Path
        path = f"../../data/instances/max_cut_{idx}_num_qubits_{num_qubits}/"
        # Load graph from path
        with open(path + "graph", 'rb') as pickle_file:
            graph = pickle.load(pickle_file)
        # Create object
        circ = Circ(graph)
        fun = circ.optimize_qaoa # optimization function

        # Append the optimal nosie free parameters as a starting point
        with open(path + f"qaoa_parameters_level_{level}", 'rb') as pickle_file:
            opt = pickle.load(pickle_file)
        if level == 1:
            x0 = opt[0]
        else:
            xmin = opt.x
            x0 = xmin
        
        # options to the minimizer
        options = {'disp': None, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-06, 'eps': 1e-05,
                   'maxfun': 500, 'maxiter': 500, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
        # lower and upper bounds
        bounds_gamma = ((0, np.pi),) * level
        bounds_beta = ((0, np.pi / 2),) * level
        bounds = bounds_gamma + bounds_beta
        res = optimize.minimize(fun, x0, args=(["DV",alpha,cutoff]), 
                                bounds=bounds, method="L-BFGS-B", options=options)

         # calculate the trace of the output
        xmin = res.x
        params = tuple(xmin[:level]), tuple(xmin[level:])
        rho = circ.simulate_qaoa(params,device="DV",amplitude=alpha,cutoff=cutoff)
        res["trace"] = np.trace(rho).real
        print("res",res)
        # Save results
        filename = f"../../data/instances/max_cut_{idx}_num_qubits_{num_qubits}/qaoa_parameters_dv_level_{level}_alpha_{alpha}_cutoff_{cutoff}"

        # Save results
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
