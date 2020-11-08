import numbers
import os
import bisect
from collections.abc import Iterable
from itertools import product, starmap, chain
from functools import partial, reduce
from operator import mul
import scipy.io as spio
import numpy as np; pi = np.pi
from qutip import *
from ptm.ptm import PTM
from ptm.ptm import *


rho_vec = qload('results/rho_vec') # initial state
for l in range(3,10):
    filename = "instance_8_"+str(l)
    mat = spio.loadmat("instances/" + filename + ".mat")
    instance = mat['instance']
    N = 8
    graph = instance['graph'][0,0]
    J = graph.todense()/2
    # Iteration level
    p = 1
    gamma = instance['gamma'][0,0][0][p-1].flatten()
    beta = instance['beta'][0,0][0][p-1].flatten()
    cv_ptm = PTM()

    temp2 = cv_ptm.rx(2*beta)
    temp1 = qload("results/"+filename+"_carb_p_1_1")

    U = 1
    for i in range(p):
        for j in range(N-1):
            for k in range(j+1,N):
                if J[j,k] != 0.:
                    U = ptm_expand_2toN(temp1, N=N, targets=[j,k]) * U
        for j in range(N):
            U = ptm_expand_1toN(temp2, N=N, target=j) * U
    rho_final = pauli_basis_to_rho(U*rho_vec)
    qsave(rho_final,"results/"+filename+"_rho_final_p_1")
    print("save complete")
