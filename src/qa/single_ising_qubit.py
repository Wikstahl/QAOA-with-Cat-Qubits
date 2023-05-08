import numpy as np
import scipy as sp
from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate
from qaoa_with_cat_qubits.cvdevice import KPOProcessor

H0 = sigmax()
H1 = sigmaz()

# Initialstate vacuum
initial_state = (basis(2,0) + basis(2,1)).unit()

def variational_state(x: tuple):
    s = initial_state
    alphas, betas = x
    for alpha, beta in zip(alphas, betas):
        s = (-1j * H0 * beta).expm() * (-1j * H1 * alpha).expm() * s
    return s


def expval(x):
    p = int(len(x) / 2)
    alphas = tuple(x[:p])
    betas = tuple(x[p:])
    x = (alphas, betas)
    return expect(H1, variational_state(x))

ranges = ((0, np.pi), (0, np.pi))
res = sp.optimize.brute(expval, ranges, Ns=100,
                        full_output=True, finish=sp.optimize.fmin)

np.save('../../data/trotter/single_ising_qubit',
        np.array(res, dtype='object'))
