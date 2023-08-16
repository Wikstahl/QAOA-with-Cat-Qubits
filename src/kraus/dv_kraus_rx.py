import numpy as np
from tqdm import tqdm
from qutip import sigmax, sigmay, sigmaz, qeye
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from forest.benchmarking.operator_tools import pauli_liouville2kraus
from qaoa_with_cat_qubits import *

d = 2  # dimension
tau = 10  # gate time
# Load the average gate fidelity
alpha = 1.36 # optimal alpha
cutoff = 20
loss_rate = 1/1500
file = np.load(f'data/average_gate_fidelity/cv_avg_fid_rx_alpha_{alpha}_cutoff_{cutoff}_gamma_{loss_rate}.npz')
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

for idx, arg in tqdm(enumerate(arg_list)):
    # Quantum circuit
    qc = QubitCircuit(1)
    qc.add_gate("RX", 0, None, arg)

     # Pauli transfer matrix
    R = np.zeros((d**2,d**2)) 
    R_ideal = np.zeros((d**2,d**2))

    # Matrix representation of the ideal quantum circuit
    U_list = qc.propagators()
    U = gate_sequence_product(U_list)

    # Create PTM
    for j in range(d**2):
        result = qp.run_state(init_state=sigma[j], qc=qc)
        Lambda = result.states[-1]
        Lambda_ideal = U * sigma[j] * U.dag()
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real
            R_ideal[i, j] = 1 / d * ((sigma[i] * Lambda_ideal).tr()).real
    # Get error channel
    R_error = R @ np.linalg.inv(R_ideal)

    # Convert choi to kraus
    kraus_operators = pauli_liouville2kraus(R_error)

    id = sum(np.conj(k.T)@k for k in kraus_operators)
    # check that the kraus sum to identity
    if np.isclose(id, np.eye(2), rtol=1e-4, atol=1e-4).all() != True:
        print(id)
        print('theta', arg)
        raise ValueError('Kraus operators must sum to identity')
    # Append kraus to list
    kraus_list.append(kraus_operators)

# Save results
file = f'data/kraus/dv_kraus_rx_alpha_{alpha}_cutoff_{cutoff}_gamma_{loss_rate}.npz'
np.savez(file, args=arg_list, kraus=np.array(kraus_list, dtype=object))
