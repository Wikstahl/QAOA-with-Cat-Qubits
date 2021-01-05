import numbers
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from numpy.random import normal

from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qip.operations import expand_operator
from qutip.qobj import Qobj
from qutip.operators import sigmaz, destroy, identity
from qutip.tensor import tensor
from qutip.qip.pulse import Pulse
from qutip.qip.noise import Noise, UserNoise

__all__ = ['QubitNoise']

class QubitNoise(UserNoise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits

    Attributes
    ----------
    t1: float or list
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list
        The indices of qubits that are acted on.
    """
    def __init__(self, N, t1=None, t2=None):
        self.N = N
        self.t1 = t1
        self.t2 = t2

    def _T_to_list(self, T, N):
        """
        Check if the relaxation time is valid

        Parameters
        ----------
        T: list of float
            The relaxation time
        N: int
            The number of component systems.

        Returns
        -------
        T: list
            The relaxation time in Python list form
        """
        if (isinstance(T, numbers.Real) and T > 0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T) == N:
            if all([isinstance(t, numbers.Real) and t > 0 for t in T]):
                return T
        else:
            raise ValueError(
                "Invalid relaxation time T={},"
                "either the length is not equal to the number of qubits, "
                "or T is not a positive number.".format(T))

    def get_noisy_dynamics(self):
        """
        Return a list of Pulse object with only trivial ideal pulse (H=0) but
        non-trivial relaxation noise.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        lindblad_noise: list of :class:`qutip.qip.Pulse`
            A list of Pulse object with only trivial ideal pulse (H=0) but
            non-trivial relaxation noise.
        """
        N = self.N
        targets = range(N)
        self.t1 = self._T_to_list(self.t1, N)
        self.t2 = self._T_to_list(self.t2, N)
        print("self.t1",self.t1)
        print("self.t2",self.t2)
        if len(self.t1) != N or len(self.t2) != N:
            raise ValueError(
                "Length of t1 or t2 does not match N, "
                "len(t1)={}, len(t2)={}".format(
                    len(self.t1), len(self.t2)))
        lindblad_noise = Pulse(None, None)

        for qu_ind in targets:
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]
            if t1 is not None:
                op = 1/np.sqrt(t1) * destroy(2)
                lindblad_noise.add_lindblad_noise(op, qu_ind, coeff=True)
            if t2 is not None:
                op = 1/np.sqrt(2*t2) * sigmaz()
                lindblad_noise.add_lindblad_noise(op, qu_ind, coeff=True)
        return lindblad_noise
