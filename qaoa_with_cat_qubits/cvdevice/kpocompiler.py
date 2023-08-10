import numpy as np
from scipy.optimize import brentq

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.compiler.gatecompiler import GateCompiler


__all__ = ['KPOCompiler']


class KPOCompiler(GateCompiler):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    num_ops: int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decomps: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, num_ops, labels):
        #super(KPOCompiler, self).__init__(N=N, params=params, num_ops = num_ops)
        self.gate_decomps = {"RZ": self.rz_dec,
                             "RX": self.rx_dec,
                             "RY": self.ry_dec,
                             "CARB": self.carb_dec}
        self.N = N
        self.num_ops = num_ops
        self.alpha = params['Coherent state']
        self.labels = labels # control labels
        self.dt = 0.01 # time step

    def decompose(self, qc):

        # TODO further improvement can be made here,
        # e.g. merge single qubit rotation gate, combine XX gates etc.
        self.coeff_list = [[] for i in range(self.num_ops)]
        self.tlist = self.N*[0] # time list for the qubits

        for gate in qc.gates:
            if gate.name not in self.gate_decomps:
                raise ValueError("Unsupported gate %s" % gate.name)
            else:
                self.gate_decomps[gate.name](gate)

        # Join sublists
        for i in range(self.num_ops):
            self.coeff_list[i] = np.array(sum(self.coeff_list[i], []))

        coeffs = np.zeros([len(self.coeff_list),len(max(self.coeff_list,key = lambda x: len(x)))])
        for i,j in enumerate(self.coeff_list):
            coeffs[i][0:len(j)] = j

        tlist = np.arange(0, len(coeffs[0])*self.dt, self.dt)
        return tlist, coeffs

    def rz_dec(self,gate):
        """
        Compiler for the RZ gate
        """
        q = gate.targets[0] # target qubit
        if gate.arg_value >= 0:
            phi = gate.arg_value % (2*np.pi) # argument
        elif gate.arg_value < 0:
            phi = gate.arg_value % (-2*np.pi) # argument
        index = self.labels.index(r"Z_%d" % q) # index of control

        # Time
        t_total = 2 # total gate time
        num_steps = int(np.ceil(t_total/self.dt)) # number of steps
        tlist = np.linspace(0, t_total, num_steps)

        # single photon pump amplitude
        def E(t,args):
            phi = args['phi']
            return np.pi*phi/(8*t_total*self.alpha)*np.sin(np.pi*t/t_total)

        # past time
        len_coeff_list = len(np.array(self.coeff_list[index]).flatten())
        t = self.dt * len_coeff_list
        if self.tlist[q] > t:
            self.coeff_list[index].append(int(np.ceil((self.tlist[q]-t)/self.dt)) * [0])
            self.coeff_list[index].append(list(E(tlist, args = {'phi': phi})))
        else:
            self.coeff_list[index].append(list(E(tlist, args = {'phi': phi})))

        self.tlist[q] = self.tlist[q] + t_total

    def ry_dec(self,gate):
        """
        Compiler for the RY gate
        """
        q = gate.targets[0] # target qubit
        phi = gate.arg_value % (2*np.pi) # argument

        # Time
        T_g = 2 # gate time of phase
        L = np.pi/2 # gate time H
        t_total = 2*L + T_g # total gate time
        num_steps = int(np.ceil(t_total/self.dt)) # number of steps
        tlist = np.linspace(0, t_total, num_steps)
        dt_list = num_steps * [self.dt]

        # two photon pump amplitude
        def G(t):
            # Square pulse of length L and amplitude a centered at (b+L/2)
            A = (np.heaviside(L - t, 0) + 2*(np.heaviside(t - L, 0) - np.heaviside(t - (T_g+L), 0)) + np.heaviside(t - (T_g+L), 0))
            return A

        # single photon pump amplitude
        def E(t,args):
            phi = args['phi']
            return np.pi*phi/(8*T_g*self.alpha)*np.sin(np.pi*(t-L)/T_g)*(np.heaviside(t-L,0)-np.heaviside(t-(T_g+L),0))

        index1 = self.labels.index((r"Y_%d" % q))
        index2 = self.labels.index((r"G_%d" % q))

        # past time
        len_coeff_list = len(np.array(self.coeff_list[index1]).flatten())
        t = self.dt * len_coeff_list
        if self.tlist[q] > t:
            zeros = int(np.ceil((self.tlist[q]-t)/self.dt)) * [0]

            self.coeff_list[index1].append(zeros + list(E(tlist, args = {'phi': phi})))
            self.coeff_list[index2].append(zeros + list(G(tlist)))
        else:
            self.coeff_list[index1].append(list(E(tlist, args = {'phi': phi})))
            self.coeff_list[index2].append(list(G(tlist)))

        self.tlist[q] = self.tlist[q] + t_total


    def rx_dec(self,gate):
        """
        Compiler for the RX gate
        """
        q = gate.targets[0] # target qubit
        theta = (gate.arg_value % (np.pi)) # argument
        t_total = 10 # total gate time
        num_steps = int(np.ceil(t_total/self.dt)) # number of steps
        tlist = np.linspace(0, t_total, num_steps)

        if self.alpha == 1:
            # polynomial coeffs
            z = np.array([-7.32159876e-01, 9.74255497e-01, 4.88620382e-01, 2.88478631e+00, -2.67318215e-03])
            p = np.poly1d(z) 
            # Define the function to find the root of
            def func(x):
                return p(x) - theta
            # Find the root using the brentq method, searching in the interval (0, some_large_value)
            Delta0 = brentq(func, 0, 1)
        elif self.alpha == 1.36:
            # polynomial coeffs
            z = np.array([-0.06546177,  0.18378162,  0.68276231,  0.93738578, -0.00126637])
            p = np.poly1d(z) 
            # Define the function to find the root of
            def func(x):
                return p(x) - theta
            # Find the root using the brentq method, searching in the interval (0, some_large_value)
            Delta0 = brentq(func, 0, 1.6)
        elif self.alpha == 2:
            # polynomial coeffs
            z = np.array([0.0014959, 0.04843742, -0.02898352, 0.06275272, -0.005201])
            p = np.poly1d(z)
            # Define the function to find the root of
            def func(x):
                return p(x) - theta
            # Find the root using the brentq method, searching in the interval (0, some_large_value)
            Delta0 = brentq(func, 0, 4.1)
        else:
            raise ValueError("This alpha is not allowed")

        # detuning
        def Delta(t,args):
            Delta0 = args['Delta0']
            return Delta0 * pow(np.sin(np.pi*t/t_total),2)

        index = self.labels.index(r"X_%d" % q)
        pulse_list = list(Delta(tlist, args = {'Delta0': Delta0}))

        # past time
        len_coeff_list = len(np.array(self.coeff_list[index]).flatten())
        t = self.dt * len_coeff_list
        if self.tlist[q] > t:
            pulse_list = int(np.ceil((self.tlist[q]-t)/self.dt)) * [0] + pulse_list
            self.coeff_list[index].append(pulse_list)
        else:
            self.coeff_list[index].append(pulse_list)

        self.tlist[q] = self.tlist[q] + t_total


    def carb_dec(self,gate):
        """
        Compiler for the CARB gate
        """
        targets = gate.targets # targets
        Theta = gate.arg_value % (2*np.pi) # argument
        t_total = 2 # total gate time
        num_steps = int(np.ceil(t_total/self.dt)) # number of steps
        tlist = np.linspace(0, t_total, num_steps)
        dt_list = num_steps * [self.dt]

        q = 0
        for i in range(targets[0]):
            if i == targets[0]:
                for j in range(i+1,targets[1]):
                    q += 1
            else:
                for j in range(i+1,self.N):
                    q += 1

        # coupling
        def g(t,args):
            Theta = args['Theta']
            A = np.pi*Theta/(8*t_total*pow(self.alpha,2)) # amplitude
            return A*np.sin(np.pi*t/t_total)

        index = self.labels.index(r"Z_%d Z_%d" % (targets[0], targets[1]))


        len_coeff_list = len(np.array(self.coeff_list[index]).flatten())
        t = self.dt*len_coeff_list
        q0 = targets[0] # qubit zero
        q1 = targets[1] # qubit one

        tmax = max([self.tlist[q0],self.tlist[q1]])

        if tmax > t:
            w = int(np.ceil((tmax-t)/self.dt))
            self.coeff_list[index].append(w*[0])
            self.coeff_list[index].append(list(g(tlist, args = {'Theta': Theta})))
        else:
            self.coeff_list[index].append(list(g(tlist, args = {'Theta': Theta})))
        self.tlist[q0] = tmax + t_total
        self.tlist[q1] = tmax + t_total
