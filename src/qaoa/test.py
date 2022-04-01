import pickle
import numpy
import multiprocessing
from cvqaoa.circ import Circ
from scipy import optimize

# define level and bounds as global variables
level = 1
# lower and upper bounds
bounds_gamma = ((0, numpy.pi),) * level
bounds_beta = ((0, numpy.pi / 2),) * level
bounds = bounds_gamma + bounds_beta

# define the optimization function as a global variable
# Path
path = "../../data/instances/max_cut_11/"
# Load graph from path
with open(path + "graph", 'rb') as pickle_file:
    graph = pickle.load(pickle_file)
# Create object
circ = Circ(graph)
fun = circ.optimize_qaoa

def minimize(x0):
    # options to the minimizer
    options = {'disp': None, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-06, 'eps': 1e-05,
               'maxfun': 500, 'maxiter': 1000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    res = optimize.minimize(fun, x0, args=(
        ["CV"]), bounds=bounds, method="L-BFGS-B", options=options)
    return res

startpoints = 2
betas = numpy.pi * numpy.random.uniform(size=(startpoints,level)) / 2
alphas = numpy.arccos(2 * numpy.random.uniform(size=(startpoints,level)) - 1)
x0 = numpy.hstack((alphas,betas))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    with multiprocessing.Pool(12) as pool:
        res_list = pool.map(minimize, x0)
    # Look for the global minimum in the list of results
    fmin = 0
    optimal_idx = -1
    for idx, elem in enumerate(res_list):
        if elem.fun < fmin:
            fmin = elem.fun
            optimal_idx = idx
    res = res_list[optimal_idx]
    print(res)
