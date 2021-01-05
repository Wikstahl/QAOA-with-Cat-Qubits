import numpy as np
from qutip import *
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import *
from qip.kpoprocessor import KPOProcessor


def test_rz():
    N = 1
    phi = np.pi/4
    qc = QubitCircuit(N = N)
    qc.add_gate("RZ", 0, None, phi)
    # create quantum processor
    kpo = KPOProcessor(N = N)
    # set up parameters
    alpha = kpo._paras['Coherent state']
    num_lvl = kpo._paras['Cut off']
    # computational basis
    cat_plus = (coherent(num_lvl, alpha) + coherent(num_lvl, -alpha)).unit()
    cat_minus = (coherent(num_lvl, alpha) - coherent(num_lvl, -alpha)).unit()
    up = (cat_plus + cat_minus)/np.sqrt(2) # logical zero
    down = (cat_plus - cat_minus)/np.sqrt(2) # logical one
    # sigma z
    sigma_z = up*up.dag() - down*down.dag()
    # initial state
    psi = (up+down).unit()
    # simulate
    result = kpo.run_state(init_state = psi, qc=qc, noisy = False)
    final_state = result.states[-1]
    # target state
    target_state = (-1j*phi/2*sigma_z).expm()*psi
    f = fidelity(final_state,target_state)
    assert abs(1-f) < 1.0e-4

def test_rx():
    N = 1
    theta = np.pi/4
    # Representation of a quantum program/algorithm, maintaining a sequence of gates.
    qc = QubitCircuit(N = N)
    qc.add_gate("RX", 0, None, theta)
    # create quantum processor
    kpo = KPOProcessor(N=N)
    # set up parameters
    alpha = kpo._paras['Coherent state']
    num_lvl = kpo._paras['Cut off']
    # computational basis
    cat_plus = (coherent(num_lvl, alpha) + coherent(num_lvl, -alpha)).unit()
    cat_minus = (coherent(num_lvl, alpha) - coherent(num_lvl, -alpha)).unit()
    up = (cat_plus + cat_minus)/np.sqrt(2)  # logical zero
    down = (cat_plus - cat_minus)/np.sqrt(2)  # logical one
    # sigma x
    sigma_x = up*down.dag() + down*up.dag()
    # initial state
    psi = (up + 1j*down).unit()
    # target state
    target_state = (-1j*theta/2*sigma_x).expm()*psi
    # simulate
    result = kpo.run_state(init_state = psi, qc=qc, noisy = False)
    final_state = result.states[-1]
    f = fidelity(final_state,target_state)
    assert abs(1-f) < 1.0e-4

def test_ry():
    N = 1
    phi = np.pi/4
    # Representation of a quantum program/algorithm, maintaining a sequence of gates.
    qc = QubitCircuit(N=N)
    qc.add_gate("RY", 0, None, phi)
    # create quantum processor
    kpo = KPOProcessor(N=N)
    # set up parameters
    alpha = kpo._paras['Coherent state']
    num_lvl = kpo._paras['Cut off']
    # computational basis
    cat_plus = (coherent(num_lvl, alpha) + coherent(num_lvl, -alpha)).unit()
    cat_minus = (coherent(num_lvl, alpha) - coherent(num_lvl, -alpha)).unit()
    up = (cat_plus + cat_minus)/np.sqrt(2)  # logical zero
    down = (cat_plus - cat_minus)/np.sqrt(2)  # logical one
    # sigma y
    sigma_y = 1j*(-up*down.dag() + down*up.dag())
    # initial state
    psi = (up+down).unit()
    # target state
    target_state = (-1j*phi/2*sigma_y).expm()*psi
    # simulate
    result = kpo.run_state(init_state=psi, qc=qc, noisy=False)
    final_state = result.states[-1]
    f = fidelity(final_state, target_state)
    assert abs(1-f) < 1.0e-4


def test_carb():
    def carb(arg_value):
        # control arbitrary phase gate
        zz = tensor(sigmaz(), sigmaz())
        return (-1j*arg_value/2*zz).expm()

    N = 2
    arg_value = np.pi
    qc = QubitCircuit(N=N)
    qc.user_gates = {"CARB": carb}
    qc.add_gate("CARB", targets=[0, 1], arg_value=arg_value)
    kpo = KPOProcessor(N=N)
    # set up parameters
    alpha = kpo._paras['Coherent state']
    num_lvl = kpo._paras['Cut off']
    eye = qeye(num_lvl)  # identity operator
    # computational basis
    cat_plus = (coherent(num_lvl, alpha) + coherent(num_lvl, -alpha)).unit()
    cat_minus = (coherent(num_lvl, alpha) - coherent(num_lvl, -alpha)).unit()
    up = (cat_plus + cat_minus)/np.sqrt(2)  # logical zero
    down = (cat_plus - cat_minus)/np.sqrt(2)  # logical one
    # sigma z
    sigma_z = ket2dm(up) - ket2dm(down)
    sigma_z1 = tensor([sigma_z, eye])
    sigma_z2 = tensor([eye, sigma_z])
    # initial state
    psi = tensor([up+down, up+down]).unit()
    # target state
    target = (-1j*arg_value/2*sigma_z1*sigma_z2).expm()*psi
    # simulate
    result = kpo.run_state(init_state=psi, qc=qc, noisy=False)
    final_state = result.states[-1]
    f = fidelity(final_state, target)
    assert abs(1-f) < 1.0e-4
