import numpy as np
from qutip import *
from qutip.qip.device.processor import Processor
from .qubitcompiler import QubitCompiler

__all__ = ['QubitProcessor']

class QubitProcessor(Processor):
    """
    The processor based on the physical implementation of
    a Transmon qubit.

    Parameters
    ----------
    N: int
        The number of qubits
    t1: float-array
        t1 times
    t2: float-array
        t2 time

    Attributes
    ----------
    params: dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser frequency, detuning etc.
    """

    def __init__(self, N, t1=None, t2=None):
        self.N = N
        self.t1 = t1
        self.t2 = t2
        super(QubitProcessor, self).__init__(self.N,
                                             t1 = self.t1,
                                             t2 = self.t2)

        # Create the control and drifts
        self.set_up_ops(N)

    def set_up_ops(self,N):
        """
        Generate the Hamiltonians and save them in the attribute `ctrls`.
        """

        for m in range(N):
            self.add_control(1/2*sigmax(),
                             targets = [m], label=r"\sigma^x_%d" % m)
            self.add_control(1/2*sigmay(),
                             targets = [m], label=r"\sigma^y_%d" % m)
            self.add_control(1/2*sigmaz(),
                             targets = [m], label=r"\sigma^z_%d" % m)

        if N > 1:
            for i in range(N-1):
                for j in range(i+1,N):
                    H_cpl = 1/2*tensor(sigmaz(),sigmaz())
                    self.add_control(H_cpl,
                                     targets = [i,j],
                                     label = (r"\sigma^z_%d\sigma^z_%d" % (i,j)))


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

        dec = QubitCompiler(N = self.N,
                            num_ops = len(self.ctrls),
                            pulses = self.pulses)
        dec.decompose(qc)

    def run_state(self, init_state=None, analytical=False, qc=None,
                  states=None, **kwargs):
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
        return super(QubitProcessor, self).run_state(
            init_state=init_state, analytical=analytical,
            states=states, **kwargs)

    def get_ops_labels(self):
        """
        Get the labels for each control Hamiltonian.
        """
        labels = []
        for i, pulse in enumerate(self.pulses):
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
