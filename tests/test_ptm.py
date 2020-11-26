import numpy as np
from qutip import *
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import *
from qip.kpoprocessor import KPOProcessor

def test_rz():
    N = 1
    phi = 2*np.pi
    qc = QubitCircuit(N = N)
    qc.add_gate("RZ", 0, None, phi)
    kpo = KPOProcessor(N = N)
    tlist, coeffs = kpo.load_circuit(qc)
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
    kpo = KPOProcessor(N = N)
    tlist, coeffs = kpo.load_circuit(qc)
    result = kpo.run_state(init_state = psi, noisy = False)
    final_state = result.states[-1]
    # target state
    target_state = (-1j*phi/2*sigma_z).expm()*psi
    f = fidelity(final_state,target_state)
    assert abs(1-f) < 1.0e-5
