from qutip import *
from qutip.qip.device import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from qaoa_with_cat_qubits import *

# KPO parameters
kpo = KPOProcessor(N=1,num_lvl=20)
alpha = kpo._paras['Coherent state']
num_lvl = kpo._paras['Cut off']

# Cat state
cat_plus = (coherent(num_lvl,alpha) + coherent(num_lvl,-alpha)).unit()
cat_minus = (coherent(num_lvl,alpha) - coherent(num_lvl,-alpha)).unit()

# Computational basis
up = (cat_plus + cat_minus)/np.sqrt(2) # Binary 0
down = (cat_plus - cat_minus)/np.sqrt(2) # Binary 1

# Pauli Matrices in computational basis
# Identity
I = up*up.dag() + down*down.dag()
# sigma z
sigma_z = up*up.dag() - down*down.dag()
# sigma x
sigma_x = up*down.dag() + down*up.dag()
# sigma y
sigma_y = 1j*(-up*down.dag() + down*up.dag())
# List with Paulis
sigma = [sigma_x, sigma_y, sigma_z]

# Init list with average gate fidelities
avg_fid = []
# List of angles
arg_list = np.linspace(0,np.pi,20)

# Loop over the list of angles and calculate the average gate fidelity
for arg in arg_list:
    # Create quantum circuit
    qc = QubitCircuit(N=1)
    qc.add_gate("RY", 0, None, arg)

    # Ideal gate
    U = (-1j*arg/2*sigma_y).expm()

    # Average Gate Fidelity
    F = 0
    for sigma_k in sigma:
        # Master equation
        result = kpo.run_state(init_state=sigma_k, qc=qc, noisy=False)
        final_state = result.states[-1]
        # Target state
        target_state = U * sigma_k * U.dag()
        F += (target_state * final_state).tr().real
    avg_fid.append(1/2 + 1/12 * F)
np.savez('../../data/average_gate_fidelity/cv_no_noise_avg_fid_ry.npz', args=arg_list, avg=avg_fid)
