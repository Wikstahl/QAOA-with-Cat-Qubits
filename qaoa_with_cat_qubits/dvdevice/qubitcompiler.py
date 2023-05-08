import numpy as np

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.compiler.gatecompiler import GateCompiler


__all__ = ['QubitCompiler']


class QubitCompiler(GateCompiler):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    global_phase: bool
        Record of the global phase change and will be returned.

    num_ops: int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decomps: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, num_ops, pulses):
        #super(QubitCompiler, self).__init__(N=N, num_ops=num_ops)
        self.gate_decomps = {"RX": self.rx_dec,
                             "RZ": self.rz_dec,
                             "CARB": self.carb_dec,
                             }
        self.N = N
        self.num_ops = num_ops
        self.pulses = pulses
        self.label_list = []
        self.tlist = [0] * N
        for pulse in self.pulses:
            pulse.tlist = np.array([0],dtype=np.double)
            pulse.coeff = np.array([],dtype=np.double)
            self.label_list.append(pulse.label)

    def decompose(self, qc):
        for gate in qc.gates:
            if gate.name not in self.gate_decomps:
                raise ValueError("Unsupported gate %s" % gate.name)
            else:
                self.gate_decomps[gate.name](gate)

        for pulse in self.pulses:
            if np.size(pulse.coeff) == 0:
                pulse.coeff = np.append(pulse.coeff, [0])
                pulse.tlist = np.append(pulse.tlist, max(self.tlist))

    def rz_dec(self, gate):
        """
        Compiler for the RZ gate
        """
        q = gate.targets[0] # target qubit
        pulse_index = self.label_list.index(r"\sigma^z_%d" % q)
        arg_value = gate.arg_value
        t_gate = 2
        coeff = arg_value/t_gate # amplitude

        t_qubit = self.tlist[q]
        t_pulse = self.pulses[pulse_index].tlist[-1]

        if t_qubit > t_pulse:
            self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, 0)
            self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, t_qubit)

        self.tlist[q] = self.tlist[q] + t_gate
        self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, coeff)
        self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, self.tlist[q])

    def rx_dec(self, gate):
        """
        Compiler for the RX gate
        """
        q = gate.targets[0] # target qubit
        pulse_index = self.label_list.index(r"\sigma^x_%d" % q) # label
        arg_value = gate.arg_value
        t_gate = 10
        coeff = arg_value/t_gate # amplitude

        t_qubit = self.tlist[q]
        t_pulse = self.pulses[pulse_index].tlist[-1]

        if t_qubit > t_pulse:
            self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, 0)
            self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, t_qubit)

        self.tlist[q] = self.tlist[q] + t_gate
        self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, coeff)
        self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, self.tlist[q])

    def carb_dec(self,gate):
        """
        Compiler for the CARB gate
        """
        targets = gate.targets # targets
        q0 = targets[0] # qubit zero
        q1 = targets[1] # qubit one
        arg_value = gate.arg_value
        t_gate = 2 # gate time
        coeff = arg_value/t_gate # amplitude

        pulse_index = self.label_list.index(r"\sigma^z_%d\sigma^z_%d"
                                            % (q0, q1)) # label

        t_qubit = max([self.tlist[q0], self.tlist[q1]]) # get past time
        t_pulse = self.pulses[pulse_index].tlist[-1]

        if t_qubit > t_pulse:
            self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, 0)
            self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, t_qubit)

        self.pulses[pulse_index].coeff = np.append(self.pulses[pulse_index].coeff, coeff)
        self.pulses[pulse_index].tlist = np.append(self.pulses[pulse_index].tlist, t_qubit + t_gate)

        self.tlist[q0] = t_qubit + t_gate
        self.tlist[q1] = t_qubit + t_gate
