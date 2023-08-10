from collections.abc import Iterable
import warnings
import numbers

import numpy as np
from qutip import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor
from .kpocompiler import KPOCompiler

__all__ = ['KPOProcessor']

class KPOProcessor(Processor):
    """
    The processor based on the physical implementation of
    a Kerr Nonlinear Resonator (KNR).
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`qutip.qip.device.ModelProcessor`)

    Parameters
    ----------
    correct_global_phase: boolean, optional
        If true, the analytical solution will track the global phase. It
        has no effect on the numerical solution.

    Attributes
    ----------
    params: dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser frequency, detuning etc.
    """

    def __init__(self, N, num_lvl=20, gamma=1/1500, alpha=2, spline_kind="cubic"):

        self.N = N # Number qubits
        self.num_lvl = num_lvl # Hilbert space dimension
        self.dims = [num_lvl] * self.N
        self.spline_kind = spline_kind
        self.alpha = alpha # Coherent state amplitude
        self.G = self.alpha**2 # Two photon drive amplitude
        self.gamma = gamma # Single photon loss rate
        self.c_ops = [] # Collapse operators

        super(KPOProcessor, self).__init__(self.N, dims=self.dims, spline_kind=self.spline_kind)

        self._paras = {}
        self.set_up_params()

        # Create the control and drifts
        self.set_up_ops(N)

    def set_up_params(self):
        """
        Save the parameters in the attribute `params`
        """
        self._paras["Coherent state"] = self.alpha
        self._paras["Cut off"] = self.num_lvl
        self._paras["Loss rate"] = self.gamma
        self._paras["TPD Amplitude"] = self.G

    @property
    def params(self):
        return self._paras

    @params.setter
    def params(self, par):
        self.set_up_params(**par)

    def set_up_ops(self,N):
        """
        Generate the Hamiltonians and save them in the attribute `ctrls`.
        """
        a = destroy(self.num_lvl)
        eye = qeye(self.num_lvl)
        H_qubits = 0

        for m in range(N):
            # Creation operator for the m:th qubit
            b = tensor([a if m == j else eye for j in range(N)])

            # Single photon drive Hamiltonian Z
            self.add_control(b.dag() + b, targets = list(range(N)), label = (r"Z_%d" % m))

            # Single photon drive Hamiltonian Y
            self.add_control(1j*(b - b.dag()), targets = list(range(N)), label = (r"Y_%d" % m))

            # Two photon drive (TPD) Hamiltonian
            self.add_control(- self.G * (b.dag()**2 + b**2), label = (r"G_%d" % m))

            # Detuning
            self.add_control(- b.dag()*b, targets = list(range(N)), label = (r"X_%d" % m))

            # Qubit Hamiltonian
            H_qubits += - b.dag()**2 * b**2 + self.G * (b.dag()**2 + b**2)

            # Collapse operators
            if self.gamma != 0:
                self.c_ops.append(np.sqrt(self.gamma) * b)

        self.add_drift(H_qubits, targets = list(range(N)))

        # Construct the Interaction/Coupling Hamiltonian
        if N > 1:
            for i in range(N-1):
                b_1 = tensor([a if i == k else eye for k in range(N)])
                for j in range(i+1,N):
                    b_2 = tensor([a if j == k else eye for k in range(N)])
                    H_cpl = b_1.dag()*b_2 + b_2.dag()*b_1
                    self.add_control(H_cpl, targets = list(range(N)), label = (r"Z_%d Z_%d" % (i,j)))

    def load_circuit(self, qc):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        labels = []
        for i, pulse in enumerate(self.pulses):
            labels.append(pulse.label)

        dec = KPOCompiler(N = self.N, params = self._paras, num_ops = len(self.ctrls), labels = labels)

        tlist, self.coeffs = dec.decompose(qc)

        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist

        return tlist, self.coeffs

    def run_state(self, init_state=None, analytical=False, qc=None,
                  states=None, noisy=False, **kwargs):
        """
        If `analytical` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.
        If `analytical` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.

        Parameters
        ----------
        init_state: Qobj
            Initial density matrix or state vector (ket).

        analytical: boolean
            If True, calculate the evolution with matrices exponentiation.

        qc: :class:`qutip.qip.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.

        states: :class:`qutip.Qobj`, optional
         Old API, same as init_state.

        **kwargs
           Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if qc is not None:
            self.load_circuit(qc)

        if init_state is None:
            init_state = tensor([coherent(self.num_lvl,self.alpha) for i in range(self.N)])

        if noisy == True:
            kwargs["c_ops"] = self.c_ops

        return super(KPOProcessor, self).run_state(
            init_state=init_state, analytical=analytical,
            states=states, **kwargs)

    def run_propagator(self, qc=None, **kwargs):
        """

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.

        states: :class:`qutip.Qobj`, optional
         Old API, same as init_state.

        **kwargs
           Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if qc is not None:
            self.load_circuit(qc)

        # construct qobjevo for unitary evolution
        noisy_qobjevo = self.get_qobjevo()

        evo_result = propagator(
            noisy_qobjevo.to_list(), noisy_qobjevo.tlist, c_op_list=self.c_ops, **kwargs)
        return evo_result

        #return super(KPOProcessor, self).run_propagator(**kwargs)

    def get_ops_labels(self):
        """
        Get the labels for each control Hamiltonian.
        """
        labels = []
        for _, pulse in enumerate(self.pulses):
            labels.append(r"$" + pulse.label + "$")
        return labels

    def get_ops_and_u(self):
        """
        Get the labels for each Hamiltonian.

        Returns
        -------
        ctrls: list
            The list of Hamiltonians
        coeffs: array_like
            The transposed pulse matrix
        """
        return (self.ctrls, self.get_full_coeffs().T)

    def pulse_matrix(self):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        -------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        #dt = 0.01
        H_ops, H_u = self.get_ops_and_u()

        # FIXME This might becomes a problem if new tlist other than
        # int the default pulses are added.
        tlist = self.get_full_tlist()
        dt = tlist[1] - tlist[0]
        t_tot = tlist[-1]
        n_t = len(tlist)
        n_ops = len(H_ops) # Number of controls = 2

        #t = np.linspace(0, t_tot, n_t) # len(t) = len(tlist)
        t = tlist
        u = H_u.T
        #u = np.zeros((n_ops, n_t))

        return tlist, u, self.get_ops_labels()

    def plot_pulses(self, title=None, figsize=(12, 6), dpi=None):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        -------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        y_shift = 0
        for n, uu in enumerate(u):
            if np.any(u[n]): # Only plot non zero pulses
                ax.plot(t,u[n],label=u_labels[n])
                ax.fill_between(t, 0, u[n], alpha=0.2)

        ax.axis('tight')
        #ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        ax.set_ylabel("Amplitude (K)")
        ax.set_xlabel("Time (1/K)")

        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax
