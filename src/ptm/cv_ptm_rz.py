import numpy as np; pi = np.pi
from qutip import *
from qutip.qip.circuit import QubitCircuit, gate_sequence_product
from qaoa_with_cat_qubits.cvdevice import KPOProcessor

## KPO parameters
alpha = 1 # amplitude of coherent state
num_lvl = 20 # hilbert space cut off
kpo = KPOProcessor(N=1,num_lvl=num_lvl,alpha=alpha)

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
# list of paulis in computational basis
sigma = [I, sigma_x, sigma_y, sigma_z]
# list of paulis in qubit basis
pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]

# Angle of rotation
arg = pi/2

# Quantum circuit
qc = QubitCircuit(1)
qc.add_gate("RZ",0,None,arg)

# Matrix representation of the ideal quantum circuit
U_list = qc.propagators()
U = gate_sequence_product(U_list)

d = 2 # dimension
# Pauli transfer matrix
R = np.zeros((d**2,d**2)) 
R_ideal = np.zeros((d**2,d**2)) 

# Compute the PTM for the noisy and ideal quantum gate
for j in range(d**2):
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
name = "ptm_error_channel_rz_alpha_"+str(alpha)+"_cutoff_"+str(num_lvl)
file = loc + name
np.save(file, error_channel, allow_pickle=True, fix_imports=True)
print("success")