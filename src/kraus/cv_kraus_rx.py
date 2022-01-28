import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from cvqaoa.cvdevice import KPOProcessor

# KPO parameters
kpo = KPOProcessor(N=1, num_lvl=20)
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

# A matrix
A = np.zeros((4, 4, 4, 4), dtype='complex')
for i in range(4):
    for j in range(4):
        for m in range(4):
            for n in range(4):
                A[i, j, m, n] = (sigma[i] * sigma[m] *
                                 sigma[j] * sigma[n]).tr()
A = A.reshape((16, 16))
A_inv = np.linalg.inv(A)

# Angle of rotation
arg_list = np.linspace(0, np.pi, num=181, endpoint=False)

# Dimension
d = 2

# Pauli transfer matrix
R = np.zeros((d**2, d**2))
R_inv = np.zeros((d**2, d**2))  # Inverse

error_channel = np.zeros((len(arg_list), d**2, d**2))

# Initialise list
kraus_list = []
for idx, arg in enumerate(arg_list):
    # Quantum circuit
    qc = QubitCircuit(1)
    qc.add_gate("RX", 0, None, arg)

    for j in range(d**2):
        result = kpo.run_state(init_state=sigma[j], qc=qc, noisy=True)
        result_inv = kpo.run_state(init_state=sigma[j], qc=qc, noisy=False)

        Lambda = result.states[-1]
        Lambda_inv = result_inv.states[-1]
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real
            R_inv[i, j] = 1 / d * ((sigma[i] * Lambda_inv).tr()).real

    # Get the PTM error channel
    ptm_error = R@np.linalg.inv(R_inv)

    # Convert PTM to choi
    rho = 1 / d**2 * sum((ptm_error[i, j] * tensor(P[i].trans(), P[j]))
                         for i in range(d**2) for j in range(d**2))
    choi = Qobj(rho.full(), dims=[[[2], [2]], [[2], [2]]], superrep='choi')

    # Convert choi to kraus. Each kraus needs to be scaled by a factor sqrt(d)
    kraus = [np.sqrt(d) * k for k in choi_to_kraus(choi)]

    # Identity
    id = sum(k * k.dag() for k in kraus)

    # Check that the kraus sum to identity
    if np.isclose(id, qeye(2), rtol=1e-5, atol=1e-5).all() != True:
        raise 'Kraus operators must sum to identity'

    # Append kraus to list
    kraus_list.append(kraus)

# Save results
file = '../../data/kraus/cv_kraus_rx.npz'
np.savez(file, args=arg_list, kraus=kraus_list)
