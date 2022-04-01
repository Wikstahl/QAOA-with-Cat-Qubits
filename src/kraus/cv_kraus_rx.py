import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from forest.benchmarking.operator_tools import *
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
    # Target
    U = (-1j*sigmax()*arg/2).expm()

    for j in range(d**2):
        result = kpo.run_state(init_state=sigma[j], qc=qc, noisy=True)
        Lambda = result.states[-1]
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real

    # Make PTM trace preserving
    R[0,:] = np.zeros(d**2)
    R[0,0] = 1

    # Convert PTM to Kraus
    kraus = pauli_liouville2kraus(R)

    # Get the kraus errorrs
    kraus_err = [k@(U.dag()).full() for k in kraus]

    # Identity
    id = sum(np.conj(k.T)@k for k in kraus_err)

    # Check that the kraus sum to identity
    if np.isclose(id, np.eye(2), rtol=1e-4, atol=1e-4).all() != True:
        print(id)
        raise 'Kraus operators must sum to identity'

    # Append kraus to list
    kraus_list.append(kraus_err)

kraus_list = np.asanyarray(kraus_list)
# Save results
file = '../../data/kraus/cv_kraus_rx.npz'
np.savez(file, args=arg_list, kraus=kraus_list)
