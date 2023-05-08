import numpy as np
from qutip import Qobj

__all__ = ['carb']

def carb(theta):
    """
    Quantum object representing the CARB gate.

    Returns
    -------
    carb_gate : qobj
        Quantum object representation of CARB gate

    Examples
    --------

    """
    return Qobj([[np.exp(-1j*theta/2), 0, 0, 0],
                 [0, np.exp(1j*theta/2), 0, 0],
                 [0, 0, np.exp(1j*theta/2), 0],
                 [0, 0, 0, np.exp(-1j*theta/2)]],
                dims=[[2, 2], [2, 2]])
