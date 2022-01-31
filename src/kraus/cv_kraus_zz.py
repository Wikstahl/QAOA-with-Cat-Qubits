import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from cvqaoa.cvdevice import KPOProcessor
from cvqaoa.gates import carb

# KPO parameters
kpo = KPOProcessor(N=2, num_lvl=20)
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
sigma = [I, sigma_x, sigma_y, sigma_z]
# qubit Paulis
P = [qeye(2), sigmax(), sigmay(), sigmaz()]

# Dimension
d = 4

# Pauli transfer matrix
R = np.zeros((d**2, d**2))
R_inv = np.zeros((d**2, d**2))  # Inverse

# Quantum circuit
qc = QubitCircuit(N=2)
qc.user_gates = {"CARB": carb}
qc.add_gate("CARB", 0, None, 0)

for i in range(d):
    for j in range(d):
        sigma_ij = tensor(sigma[i], sigma[j])
        result = kpo.run_state(init_state=sigma_ij, qc=qc, noisy=True)
        result_inv = kpo.run_state(init_state=sigma_ij, qc=qc, noisy=False)

        Lambda = result.states[-1]
        Lambda_inv = result_inv.states[-1]

        for k in range(d):
            for l in range(d):
                sigma_kl = tensor(sigma[k], sigma[l])
                R[(k + l), (i + j)] = 1 / d * ((sigma_kl * Lambda).tr()).real
                R_inv[(k + l), (i + j)] = 1 / d * \
                    ((sigma_kl * Lambda_inv).tr()).real

# Get the PTM error channel
ptm_error = R@np.linalg.inv(R_inv)

# Convert PTM to choi
rho = np.zeros((d**2, d**2))
for i in range(d):
    for j in range(d):
        P_ij = tensor(P[i], P[j])
        for k in range(d):
            for l in range(d):
                rho[(i + j), (k + l)] = 1 / d**2 * \
                    sum(ptm_error[(i + j), (k + l)]
                        * tensor(P_ij.trans(), P_kl))

choi = Qobj(rho, dims=[[[2, 2], [2, 2]],
                       [[2, 2], [2, 2]]], superrep='choi')

# Convert choi to kraus. Each kraus needs to be scaled by a factor sqrt(d)
kraus = [np.sqrt(d) * k for k in choi_to_kraus(choi)]

# Identity
id = sum(k * k.dag() for k in kraus)

# Check that the kraus sum to identity
if np.isclose(id.full(), qeye(4).full(), rtol=1e-5, atol=1e-5).all() != True:
    raise 'Kraus operators must sum to identity'

# Save results
file = '../../data/kraus/cv_kraus_zz.npz'
np.save(file, kraus)
