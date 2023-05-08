import numpy as np
import scipy as sp
from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate
from qaoa_with_cat_qubits.cvdevice import KPOProcessor

num_lvl = 20  # Hilbert space cut-off
a = destroy(num_lvl)  # Annihilation operator
eye = qeye(num_lvl)  # Identity operator

# Parameters
K = 1  # Kerr amplitude
G = 4  # Two-photon drive amplitude (in units of K)
alpha = np.sqrt(G / K)  # Coherent state amplitude
Delta = 1 / (alpha**2 * np.exp(-2 * alpha**2))  # Detuning (in units of K)

# Cat state
cat_plus = (coherent(num_lvl, alpha) + coherent(num_lvl, -alpha)).unit()
cat_minus = (coherent(num_lvl, alpha) - coherent(num_lvl, -alpha)).unit()

# Computational basis
up = (cat_plus + cat_minus) / np.sqrt(2)  # Binary 0
down = (cat_plus - cat_minus) / np.sqrt(2)  # Binary 1

# Pauli Matrices in computational basis
# Identity
I = up * up.dag() + down * down.dag()
# sigma z
sigma_z = up * up.dag() - down * down.dag()
# sigma x
sigma_x = up * down.dag() + down * up.dag()
# sigma y
sigma_y = 1j * (-up * down.dag() + down * up.dag())
# List with Paulis
sigma = [sigma_x, sigma_y, sigma_z]

# Mixer Hamiltonian
H0 = - Delta * a.dag() * a - K * a.dag()**2 * a**2
# Cost Hamiltonian
H1 = - K * a.dag()**2 * a**2 + G * (a.dag()**2 + a**2)
# Cost Hamiltonian computational basis
HC = -ket2dm(cat_plus) # Use minus to make it minimization

# Initialstate vacuum
initial_state = basis(num_lvl, 0)

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
    return expect(HC, variational_state(x))


ranges = ((0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi))
res = sp.optimize.brute(expval, ranges, Ns=100,
                        full_output=True, finish=sp.optimize.fmin)

np.save('../../data/trotter/cat_state_2',
        np.array(res, dtype='object'))
