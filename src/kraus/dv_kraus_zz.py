# Good Sourde https://forest-benchmarking.readthedocs.io/en/latest/superoperator_representations.html
import numpy as np
import scipy as sp
from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from qaoa_with_cat_qubits import *
from qaoa_with_cat_qubits.gates import carb
from forest.benchmarking.operator_tools import *
from forest.benchmarking.operator_tools import *
from forest.benchmarking.operator_tools.project_superoperators import *


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
alpha = 1.36 # optimal alpha
cutoff = 20
loss_rate = 1/1500
file = np.load(f'data/average_gate_fidelity/cv_avg_fid_rzz_alpha_{alpha}_cutoff_{cutoff}_gamma_{loss_rate}.npz')
f_bar = np.mean(file['avg'])
# Find the corresponding T1
gamma = (d + 1) / (d * tau) * (1 - f_bar)
T1 = 1 / (gamma)
# Qubit Processor
qp = QubitProcessor(N=N, T1=T1)

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

    # Quantum circuit
    qc = QubitCircuit(N)
    qc.user_gates = {"CARB": carb}
    qc.add_gate("CARB", targets=[0, 1], arg_value=arg)

    # Target
    U = (-1j*tensor(sigmaz(),sigmaz())*arg/2).expm()

    # Create PTM
    opt = Options(nsteps=1e6)
    for j in range(d**2):
        eVals, eVecs = np.linalg.eigh(sigma[j])
        Lambda_tot = 0
        for idx, eVal in enumerate(eVals):
            # Initial state
            init_state = ket2dm(Qobj(eVecs[:,idx],dims=[[2,2],[1,1]]))
            # Evolve state
            result = qp.run_state(init_state=init_state, qc=qc, options=opt)
            Lambda_tot += eVal * result.states[-1]
        for i in range(d**2):
            R[i, j] = 1 / d * ((sigma[i] * Lambda_tot).tr()).real

    # Convert choi to kraus
    kraus = choi2kraus(choi)
    kraus_err = [np.sqrt(d)*k@(U.dag()).full() for k in kraus]
    id = sum(np.conj(k.T)@k for k in kraus_err)

    # check that the kraus sum to identity
    if np.isclose(id, np.eye(4), rtol=1e-6, atol=1e-3).all() != True:
        print(id)
        print('theta', arg)
        raise 'Kraus operators must sum to identity'
    # Append kraus to list
    kraus_list.append(kraus_err)

# Save results
file = f'data/kraus/dv_kraus_rzz_alpha_{alpha}_cutoff_{cutoff}_gamma_{loss_rate}.npz'
np.savez(file, args=arg_list, kraus=np.array(kraus_list, dtype=object))
