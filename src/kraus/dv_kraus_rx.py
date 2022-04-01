import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from forest.benchmarking.operator_tools import *
from cvqaoa import *

d = 2  # dimension
tau = 10  # gate time
# Load the average gate fidelity
file = np.load('../../data/average_gate_fidelity/cv_avg_fid_rx.npz')
f_bar = np.mean(file['avg'])
# Find the corresponding T1
gamma = 2 * (d + 1) / (d * tau) * (1 - f_bar)
T1 = 1 / (gamma)

# Qubit Processor
qp = QubitProcessor(N=1, T1=T1)
qp_inv = QubitProcessor(N=1, T1=None)

# Pauli matrices
sigma = [qeye(2), sigmax(), sigmay(), sigmaz()]

# List with angles of rotation
arg_list = np.linspace(0, np.pi, num=181, endpoint=False)

# Initialize kraus list
kraus_list = []

for idx, arg in enumerate(arg_list):
    # Pauli transfer matrix
    R = np.zeros((d**2, d**2))

    # Quantum circuit
    qc = QubitCircuit(1)
    qc.add_gate("RX", 0, None, arg)

    # Target
    U = (-1j*sigmax()*arg/2).expm()

    # Create PTM
    for j in range(d**2):
        result = qp.run_state(init_state=sigma[j], qc=qc)

        Lambda = result.states[-1]
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real

    # Convert to kraus
    kraus = pauli_liouville2kraus(R)
    kraus_err = [k@(U.dag()).full() for k in kraus]
    id = sum(np.conj(k.T) @ k for k in kraus)

    # check that the kraus sum to identity
    if np.isclose(id, np.eye(2), rtol=1e-4, atol=1e-4).all() != True:
        print(id)
        raise 'Kraus operators must sum to identity'
    # Append kraus to list
    kraus_list.append(kraus_err)

# Save results
file = '../../data/kraus/dv_kraus_rx.npz'
np.savez(file, args=arg_list, kraus=np.array(kraus_list, dtype=object))
