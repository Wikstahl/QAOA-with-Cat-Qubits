import numpy as np
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from cvqaoa import *
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


N = 2  # qubits
d = 4  # dimension
tau = 2  # gate time
# Load the average gate fidelity
file = np.load('../../data/average_gate_fidelity/cv_avg_fid_zz.npz')
f_bar = np.mean(file['avg'])
# Find the corresponding T1
gamma = (d + 1) / (d * tau) * (1 - f_bar)
T1 = 1 / (gamma)
# Qubit Processor
qp = QubitProcessor(N=N, T1=T1)
qp_inv = QubitProcessor(N=N, T1=None)

# Pauli matrices
P = [qeye(2), sigmax(), sigmay(), sigmaz()]

# Create all tensor products
sigma = list(map(tensor, product(P, repeat=2)))

# List with angles of rotation
arg_list = np.linspace(0, np.pi, num=181, endpoint=False)

# Initialize kraus list
kraus_list = []

for idx, arg in enumerate(arg_list):
    # Pauli transfer matrix
    R = np.zeros((d**2, d**2))
    R_inv = np.zeros((d**2, d**2))  # inverse/ideal

    # Quantum circuit
    qc = QubitCircuit(N)
    qc.user_gates = {"CARB": carb}
    qc.add_gate("CARB", targets=[0, 1], arg_value=arg)

    # Quantum circuit inverse
    qc_inv = QubitCircuit(N)
    qc_inv.user_gates = {"CARB": carb}
    qc_inv.add_gate("CARB", targets=[0, 1], arg_value=arg)

    # Create PTM
    opt = Options(nsteps=1e6)
    for j in range(d**2):
        result = qp.run_state(init_state=sigma[j], qc=qc, options=opt)
        result_inv = qp_inv.run_state(
            init_state=sigma[j], qc=qc_inv, options=opt)

        Lambda = result.states[-1]
        Lambda_inv = result_inv.states[-1]
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda).tr()).real
            R_inv[i, j] = 1 / d * ((sigma[i] * Lambda_inv).tr()).real

    # Find the PTM of the error channel
    ptm_error = R@np.linalg.inv(R_inv)
    # Convert PTM to choi
    choi = 1 / d**2 * sum((ptm_error[i, j] * tensor(sigma[j].trans(), sigma[i]))
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
        print(id)
        print('theta', arg)
        raise 'Kraus operators must sum to identity'
    # Append kraus to list
    kraus_list.append(kraus)

# Save results
file = '../../data/kraus/dv_kraus_zz.npz'
np.savez(file, args=arg_list, kraus=np.array(kraus_list, dtype=object))
