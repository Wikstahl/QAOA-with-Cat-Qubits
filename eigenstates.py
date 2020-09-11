#%matplotlib
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from qutip import *

name = 'data/instance_5_1/instance_5_1'
mat = spio.loadmat(name + '.mat')
instance = mat['instance']
eigvals = instance['eigvals']
num_q = instance['size'][0][0][0][0]
J = instance['J'][0][0]
h = np.array(instance['h'][0][0]).flatten()
costs = np.array(eigvals[0][0]).flatten()

num_lvl = 12 # number of levels in Hilbert space
K = 1 # amplitude of Kerr nonlinearity
G = 4 # amplitude of two-photon drive
alpha = pow(G/K,1/2) # amplitude of coherent state
F = h/(2 * alpha) # amplitude of single-photon drives
delta = 0.5 # amplitude of dephasing

a = destroy(num_lvl) # annihilation operator
eye = qeye(num_lvl) # identity operator

# Computational basis
up = coherent(num_lvl,alpha) # |↑⟩ → |0⟩
down = coherent(num_lvl,-alpha) # |↓⟩ → |1⟩

# Initial Hamiltonian
H_i = 0
# Final Hamiltonian
H_p = 0
for q in range(num_q):
    b = tensor([a if q==j else eye for j in range(num_q)]) # annihilation operator for the q:th qubit
    H_i += - delta * b.dag() * b - K * pow(b.dag(),2) * pow(b,2)
    H_p += - K * pow(b.dag(),2) * pow(b,2) + G * (pow(b.dag(),2) + pow(b,2)) + F[q] * (b.dag() + b)
    if q < (num_q-1) and J[q][q+1] != 0:
        c = tensor([a if (q+1)==j else eye for j in range(num_q)]) # annihilation operator for the q:th qubit
        H_int = J[q][q+1]/pow(alpha,2) * (b.dag() * c + c.dag() * b)
        H_i += H_int
        H_p += H_int

# time dependent factor for the initial Hamiltonian
def H_i_coeff(t, args):
    tau = args['tau']
    return (1 - t/tau)

# time dependent factor for the final Hamiltonian
def H_p_coeff(t,args):
    tau = args['tau']
    return t/tau

# total Hamiltonian
H_tot = [[H_i,H_i_coeff],[H_p, H_p_coeff]]

slist = np.linspace(0,1,51)
eigenenergies = []
eigenvectors = []
for i,s in enumerate(slist):
    H = (1-s)*H_i + s*H_p
    eigenstates = H.eigenstates(sparse=True, sort='high', eigvals=2, tol=0, maxiter=100000)
    eigenenergies.append(eigenstates[0])
    eigenvectors.append(eigenstates[1])
eigenenergies = np.array(eigenenergies)
eigenenergies.tolist();

qsave(eigenenergies,'data/instance_5_1/eigenenergies')
qsave(eigenvectors,'data/instance_5_1/eigenvectors')
