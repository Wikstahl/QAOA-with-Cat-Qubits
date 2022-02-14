import pickle
import numpy as np
import networkx as nx
from scipy.optimize import brute
from cvqaoa.circ import Circ

# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/instances/max_cut_"+str(idx)+"/"
    # Load graph
    with open(path + "graph", 'rb') as pickle_file:
        graph = pickle.load(pickle_file)
    # Create object
    circ = Circ(graph)
    # Optimization bounds
    ranges = ((0, np.pi), (0, np.pi/2))
    # Brute force on 100 x 100 grid
    res = brute(circ.optimize_qaoa, ranges, args=(["CV"]), Ns=100,
                full_output=True, finish=None, workers=-1)
    # Save results
    filename = path + "qaoa_parameters_cv"
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
