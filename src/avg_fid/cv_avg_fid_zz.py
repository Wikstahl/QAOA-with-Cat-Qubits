import numpy as np
from qutip import *
from qutip.qip.device import *
from tqdm import tqdm
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit
from qaoa_with_cat_qubits import KPOProcessor
from qaoa_with_cat_qubits.gates import carb

# KPO parameters
cutoff = 20
alpha = 1.36
gamma = 1/1500
kpo = KPOProcessor(N=2,num_lvl=20,alpha=alpha,gamma=gamma)
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
sigma = [I, sigma_x, sigma_y, sigma_z]

# Init list with average gate fidelities
avg_fid = []
# List of angles
arg_list = np.linspace(0,np.pi,20)

# Loop over the list of angles and calculate the average gate fidelity
for i, arg in tqdm(enumerate(arg_list)):
    # Create quantum circuit
    qc = QubitCircuit(N=2)
    qc.user_gates = {"CARB": carb}
    qc.add_gate("CARB", targets = [0,1], arg_value = arg)

    # Ideal gate
    U = (-1j*arg*tensor(sigma_z,sigma_z)/2).expm()

    # Average Gate Fidelity
    d = 4
    F = 0
    for sigma_k in sigma:
        for sigma_l in sigma:
            sigma_kl = tensor(sigma_k,sigma_l)
            # Master equation
            result = kpo.run_state(init_state=sigma_kl, qc=qc, noisy=True)
            final_state = result.states[-1]
            # Target state
            target_state = U * sigma_kl.dag() * U.dag()
            F += (target_state * final_state).tr().real
    avg_fid.append((F + d**2) / (d**2*(d+1)))
    print(avg_fid)
np.savez(f'cv_avg_fid_zz_alpha_{alpha}_cutoff_{cutoff}_gamma_{gamma}.npz', args=arg_list, avg=avg_fid)
