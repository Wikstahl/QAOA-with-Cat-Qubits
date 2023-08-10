import numpy as np; pi = np.pi
from qutip import *
from tqdm import tqdm
from qutip.qip.circuit import QubitCircuit, gate_sequence_product
from qaoa_with_cat_qubits.cvdevice import KPOProcessor

def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield list(prod)

## KPO parameters
alpha = 1 # amplitude of coherent state
num_lvl = 20 # hilbert space cut off
N = 2  # number of qubits
kpo = KPOProcessor(N=N,num_lvl=num_lvl,alpha=alpha)

## Cat state
cat_plus = (coherent(num_lvl,alpha) + coherent(num_lvl,-alpha)).unit()
cat_minus = (coherent(num_lvl,alpha) - coherent(num_lvl,-alpha)).unit()

## Computational basis
up = (cat_plus + cat_minus)/np.sqrt(2) # Binary 0
down = (cat_plus - cat_minus)/np.sqrt(2) # Binary 1

## Pauli Matrices in computational basis
# Identity
I = up*up.dag() + down*down.dag()
# sigma z
sigma_z = up*up.dag() - down*down.dag()
# sigma x
sigma_x = up*down.dag() + down*up.dag()
# sigma y
sigma_y = 1j*(-up*down.dag() + down*up.dag())

# List with Pauli matrices
sigma = list(
    map(tensor, product([I, sigma_x, sigma_y, sigma_z], repeat=2)))

# Create all qubit paulis
pauli = list(
    map(tensor, product([qeye(2), sigmax(), sigmay(), sigmaz()], repeat=2)))

# Angle of rotation
arg = pi/2

# Define the Ising-ZZ gate
def carb(arg_value):
    # control arbitrary phase gate
    zz = tensor(sigmaz(),sigmaz())
    return (-1j*arg_value/2*zz).expm()

# Quantum circuit
qc = QubitCircuit(N=N)
qc.user_gates = {"CARB": carb}
qc.add_gate("CARB", targets=[0, 1], arg_value=arg)

# Matrix representation of the ideal quantum circuit
U_list = qc.propagators()
U = gate_sequence_product(U_list)

d = 2**N # dimension
# Pauli transfer matrix
R = np.zeros((d**2,d**2)) 
R_ideal = np.zeros((d**2,d**2)) 

# Compute the PTM for the noisy and ideal quantum gate
for j in tqdm(range(d**2)):
    result = kpo.run_state(init_state=sigma[j],qc=qc,noisy=True)
    Lambda = result.states[-1]
    Lambda_ideal = U * pauli[j] * U.dag()
    for i in range(d**2):
        R[i,j] = 1/d * ((sigma[i]*Lambda).tr()).real
        R_ideal[i,j] = 1/d * ((pauli[i]*Lambda_ideal).tr()).real

# Extract the error channel 
error_channel = R @ np.linalg.inv(R_ideal)

# Save the result
loc = "data/ptm/" # location
name = "ptm_error_channel_rzz_alpha_"+str(alpha)+"_cutoff_"+str(num_lvl)
file = loc + name
np.save(file, error_channel, allow_pickle=True, fix_imports=True)
print("success")