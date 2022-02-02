import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from cvqaoa.cvdevice import KPOProcessor
from cvqaoa.gates import carb


def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield list(prod)


# KPO parameters
N = 2  # number of qubits
kpo = KPOProcessor(N=N, num_lvl=20)
alpha = kpo._paras['Coherent state']
num_lvl = kpo._paras['Cut off']

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
# List with Pauli matrices
sigma = list(
    map(tensor, product([I, sigma_x, sigma_y, sigma_z], repeat=2)))

# Create all qubit paulis
P = list(
    map(tensor, product([qeye(2), sigmax(), sigmay(), sigmaz()], repeat=2)))

# Dimension
d = 2**N

# Pauli transfer matrix
R = np.zeros((d**2, d**2))
R_inv = np.zeros((d**2, d**2))  # Inverse

# Quantum circuit
qc = QubitCircuit(N=N)
qc.user_gates = {"CARB": carb}
qc.add_gate("CARB", targets=[0, 1], arg_value=0)

for j in range(d**2):
    result = kpo.run_state(init_state=sigma[j], qc=qc, noisy=True)
    result_inv = kpo.run_state(init_state=sigma[j], qc=qc, noisy=False)

    Lambda = result.states[-1]
    Lambda_inv = result_inv.states[-1]

    for i in range(d**2):
        R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real
        R_inv[i, j] = 1 / d * ((sigma[i] * Lambda_inv).tr()).real

# Find the PTM of the error channel
ptm_error = R@np.linalg.inv(R_inv)
# Convert PTM to choi
choi = 1 / d**2 * sum((ptm_error[i, j] * tensor(P[j].trans(), P[i]))
                      for i in range(d**2) for j in range(d**2))
choi.dims = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]
choi.superrep = 'choi'

# Find eigenvectors and eigenvalues to choi
eValues, eVectors = np.linalg.eigh(choi)
# Because of machine impressision we drop terms smaller than rtol
rtol = 1e-6
idx, = np.where(eValues < rtol)
eVectors = np.delete(eVectors, idx, axis=1)  # drop columns
eValues = np.delete(eValues, idx)
num = len(eValues)
# Get the Kraus operators
kraus = [np.sqrt(d * eValues[i]) * eVectors[:, i].reshape((d, d))
         for i in range(num)]

# identity
id = sum(k@np.conj(k.T) for k in kraus)
# check that the kraus sum to identity
if np.isclose(id, np.eye(4), rtol=1e-3, atol=1e-3).all() != True:
    print('Kraus operators must sum to identity')
    #raise 'Kraus operators must sum to identity'

# Save results
file = '../../data/kraus/cv_kraus_zz'
np.save(file, kraus)
