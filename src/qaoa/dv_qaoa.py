import pickle
import numpy as np
from os.path import exists
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
level = 5

# Loop over all instances
for idx in range(30):
    print(idx)
    # Path
    path = "../../data/instances/max_cut_" + str(idx) + "/"
    # Load graph from path
    with open(path + "graph", 'rb') as pickle_file:
        graph = pickle.load(pickle_file)
    # Create object
    circ = Circ(graph)
    fun = circ.optimize_qaoa # optimization function
    if level == 1:
        # Optimization bounds
        ranges = ((0, np.pi), (0, np.pi / 2))
        # Brute force on 100 x 100 grid
        res = brute(circ.optimize_brute_qaoa, ranges, args=(["DV"]), Ns=100,
                    full_output=True, finish=None, workers=-1)
        # Save results
        filename = path + f"qaoa_parameters_dv_level_{level}"
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
    # Use Multistart for level > 1
    if level > 1:
        # lower and upper bounds
        bounds_gamma = ((0, np.pi),) * level
        bounds_beta = ((0, np.pi / 2),) * level
        bounds = bounds_gamma + bounds_beta

        def minimize(x0):
            # options to the minimizer
            options = {'disp': None, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-06, 'eps': 1e-05,
                       'maxfun': 500, 'maxiter': 500, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
            res = optimize.minimize(fun, x0, args=(
                ["DV"]), bounds=bounds, method="L-BFGS-B", options=options)
            return res

        startpoints = 1# 50 * level
        betas = np.pi * np.random.uniform(size=(startpoints,level)) / 2
        alphas = np.arccos(2 * np.random.uniform(size=(startpoints,level)) - 1)
        # Append the optimal nosie free parameters as a starting point
        with open(path + f"qaoa_parameters_level_{level}", 'rb') as pickle_file:
            opt = pickle.load(pickle_file)
        xmin = opt.x
        #x0 = xmin
        x0 = np.hstack((alphas,betas))
        x0 = np.vstack((x0, xmin))

        with open(path +  f"qaoa_parameters_dv_level_{level-1}", 'rb') as pickle_file:
            prev_res = pickle.load(pickle_file)
        xGuess = interpolation(prev_res.x)
        x0 = np.vstack((x0,xGuess))

        #x0 = np.hstack((alphas,betas))

        if __name__ == '__main__':
            multiprocessing.freeze_support()
            with multiprocessing.Pool(6) as pool:
                res_list = pool.map(minimize, x0)
            # Look for the global minimum in the list of results
            fmin = 0
            optimal_idx = -1
            for idx, elem in enumerate(res_list):
                if elem.fun < fmin:
                    fmin = elem.fun
                    optimal_idx = idx
            # this is the best result
            res = res_list[optimal_idx]
            # calculate the trace of the output
            xmin = res.x
            params = tuple(xmin[:level]), tuple(xmin[level:])
            rho = circ.simulate_qaoa(params,device="DV")
            res["trace"] = np.trace(rho).real

            # Save results
            filename = path + f"qaoa_parameters_dv_level_{level}"
            with open(filename, 'wb') as f:
                pickle.dump(res, f)
