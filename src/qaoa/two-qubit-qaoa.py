import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np; pi = np.pi

from qutip import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import *
from cvqaoa.cvdevice import KPOProcessor
from scipy.optimize import minimize
from matplotlib import cm

"""
Initialize problem
"""
# Problem
J = -1*np.array([[1,1/2],[1/2,1/2]])
h = -1*np.array([1/2,0])

"""
Setup parameters
"""
N = 2 # Number of qubits in the system
kpo = KPOProcessor(N = N)
alpha = kpo._paras['Coherent state']
num_lvl = kpo._paras['Cut off']
eye = qeye(num_lvl)

# Cat state
cat_plus = (coherent(num_lvl,alpha) + coherent(num_lvl,-alpha)).unit()
cat_minus = (coherent(num_lvl,alpha) - coherent(num_lvl,-alpha)).unit()

# computational basis
up = (cat_plus + cat_minus)/np.sqrt(2) # Binary 0
down = (cat_plus - cat_minus)/np.sqrt(2) # Binary 1

# pauli-z
sigma_z = ket2dm(up) - ket2dm(down)
identity = ket2dm(up) + ket2dm(down)

# possible initial states
plus = cat_plus
iplus = (cat_plus + 1j*cat_minus)/np.sqrt(2)

# initial state
initial_state = tensor([plus for i in range(N)])

# Create Hamiltonian
H = - J[0,1]*tensor(sigma_z,sigma_z) - h[0]*tensor(sigma_z,identity)

# Construct Ising-ZZ gate
def carb(arg_value):
    # control arbitrary phase gate
    zz = tensor(sigmaz(),sigmaz())
    return (-1j*arg_value/2*zz).expm()


# Define QAOA circuit
def qaoa_circuit(params):
    alphas, betas = params

    # Representation of a quantum program/algorithm, maintaining a sequence of gates.
    qc = QubitCircuit(N = N, reverse_states = False)
    qc.user_gates = {"CARB": carb}
    qc.add_state(state = "+", targets = range(N), state_type = "input")

    for alpha,beta in zip(alphas,betas):
        for j in range(N-1):
            for k in range(j+1,N):
                if J[j][k] != 0:
                    qc.add_gate("CARB", targets = [j,k], arg_value = 2*alpha*J[j][k])
        for j in range(N):
            if h[j] != 0:
                qc.add_gate("RZ", j, None, 2*alpha*h[j])
            if 2*beta > np.pi:
                qc.add_gate("RX", j, None, 2*np.pi - 2*beta)
    return qc

# Create cost function
def cost_fun(x):
    lvl = int(len(x)/2)
    alphas = x[:lvl]
    betas = x[lvl:]
    params = (alphas, betas)
    # simulate
    result = kpo.run_state(init_state = initial_state, qc = qaoa_circuit(params), noisy = True)
    final_state = result.states[-1]
    if final_state.type == 'ket':
        final_state = ket2dm(final_state)
    # compute cost
    return (H*final_state).tr().real

x0 = [0.9046, 2.6893] # Use no noisy angles as initial guess
res = minimize(cost_fun, x0=x0)

np.save('two-qubit-qaoa-1.npy', res)

qaoa_level = 1
xmin = res.x
alphas = xmin[:qaoa_level]
betas = xmin[qaoa_level:]
result = kpo.run_state(init_state = initial_state, qc = qaoa_circuit(params), noisy = True)

final_state = result.states[-1]
qsave(final_state,'noisy-qaoa-level-1-rx-mixer')

target_state = tensor(down,up)
if final_state.type == 'ket':
    f = (target_state.dag() * final_state).full()[0][0]
    f = abs(f)**2
else:
    f = (ket2dm(target_state) * final_state).tr().real
print("Success Probability = %s %%" % round(f*100,4))
