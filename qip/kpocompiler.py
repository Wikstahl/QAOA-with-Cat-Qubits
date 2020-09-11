import numpy as np
import bisect

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
        super(KPOCompiler, self).__init__(
            N=N, params=params, num_ops=num_ops)
        self.gate_decomps = {"RZ": self.rz_dec,
                             "RX": self.rx_dec,
                             "RY": self.ry_dec,
                             "CARB": self.carb_dec}
        self.N = N
        self.alpha = self.params['Coherent state']
        self.labels = labels

    def decompose(self, qc):

        def qc_depth(qc):
            """
            Returns:
                int: Depth of circuit.

            Notes:
                Based on this reference
                https://quantumcomputing.stackexchange.com/questions/5769/how-to-calculate-circuit-depth-properly
            """
            N = qc.N
            levels = [0] * N
            gates = qc.gates
            for gate in gates:
                targets = []
                for target in gate.targets:
                    levels[target] += 1
                    targets.append(target)
                if gate.controls is not None:
                    for control in gate.controls:
                        levels[control] += 1
                        targets.append(control)
            max_level = max([levels[i] for i in targets])
            return max_level

        def qc_layers(self):
            """
            Returns:
                int: Layers
            """
            layer = [[] for i in range(self.depth())] # replace this one
            N = self.N
            levels = [0] * N
            gates = self.gates
            for gate in gates:
                targets = []
                for target in gate.targets:
                    levels[target] += 1
                    targets.append(target)
                if gate.controls is not None:
                    for control in gate.controls:
                        levels[control] += 1
                        targets.append(control)
                idx = max([levels[i] for i in targets])
                for target in targets:
                    if levels[target] < idx:
                        levels[target] = idx
                idx = max([levels[i] for i in targets])-1
                layer[idx].append(gate)
            return layer

        # TODO further improvement can be made here,
        # e.g. merge single qubit rotation gate, combine XX gates etc.
        layers = qc_layers(qc)

        self.dt_list = [[] for i in range(len(layers))]
        self.coeff_list = [[] for i in range(self.num_ops)]

        temp = []

        for layer_idx, layer in enumerate(layers):
            for gate in layer:
                if gate.name not in self.gate_decomps:
                    raise ValueError("Unsupported gate %s" % gate.name)
                else:
                    self.gate_decomps[gate.name](gate,layer_idx)
            coeff_len = 0

            for i in range(self.num_ops):
                try:
                    if len(self.coeff_list[i][layer_idx]) > coeff_len:
                        coeff_len = len(self.coeff_list[i][layer_idx])
                except:
                    0

            for i in range(self.num_ops):
                try:
                    if len(self.coeff_list[i][layer_idx]) < coeff_len:
                        self.coeff_list[i][layer_idx].extend([0] * (coeff_len-len(self.coeff_list[i][layer_idx])))
                except:
                    self.coeff_list[i].append([0] * coeff_len)

            try:
                temp.extend(max(self.dt_list[layer_idx], key = len)) # this is the time for layer 1
            except:
                0

        tlist = np.empty(len(temp))
        t = 0
        for i in range(len(temp)):
            tlist[i] = t
            t += temp[i]

        # Join sublists
        for i in range(self.num_ops):
            self.coeff_list[i] = np.array(sum(self.coeff_list[i], []))

        coeffs = self.coeff_list
        return tlist, coeffs

    def rz_dec(self,gate,idx):
        """
        Compiler for the RZ gate
        """
        q = gate.targets[0] # target qubit
        phi = gate.arg_value % (2*np.pi) # argument
        t_total = 2 # total gate time

        # single photon pump amplitude
        def E(t,args):
            phi = args['phi']
            return np.pi*phi/(8*t_total*self.alpha)*np.sin(np.pi*t/t_total)

        tlist = np.linspace(0,t_total,t_total*10+1)
        dt_list = tlist[1:] - tlist[:-1]
        dt_list = np.append(dt_list,dt_list[0])
        self.dt_list[idx].append(dt_list)

        index = self.labels.index("\sigma^z_%d" % q)
        self.coeff_list[index].append(list(E(tlist, args = {'phi': phi})))

    def ry_dec(self,gate,idx):
        """
        Compiler for the RY gate
        """
        q = gate.targets[0] # target qubit
        phi = gate.arg_value % (np.pi) # argument
        T_g = 2 # gate time of phase
        L = np.pi/2 # gate time H
        t_total = 2*L + T_g # total gate time

        # two photon pump amplitude
        def G(t):
            # Square pulse of length L and amplitude a centered at (b+L/2)
            A = (np.heaviside(L - t, 0) + 2*(np.heaviside(t - L, 0) - np.heaviside(t - (T_g+L), 0)) + np.heaviside(t - (T_g+L), 0))
            return A

        # single photon pump amplitude
        def E(t,args):
            phi = args['phi']
            return np.pi*phi/(8*T_g*self.alpha)*np.sin(np.pi*(t-L)/T_g)*(np.heaviside(t-L,0)-np.heaviside(t-(T_g+L),0))

        tlist = np.linspace(0,t_total,500)
        dt_list = tlist[1:] - tlist[:-1]
        dt_list = np.append(dt_list,dt_list[0])
        self.dt_list[idx].append(dt_list)

        index1 = self.labels.index(("\sigma^y_%d" % q))
        index2 = self.labels.index(("F_%d" % q))
        self.coeff_list[index1].append(list(E(tlist, args = {'phi': phi})))
        self.coeff_list[index2].append(list(G(tlist)))

    def rx_dec(self,gate,idx):
        """
        Compiler for the RX gate
        """
        q = gate.targets[0] # target qubit
        theta = (gate.arg_value % (-np.pi)) # argument
        t_total = 10 # total gate time

        theta_list =   [0.0,
                        0.0,
                        -0.01755079694742901,
                        -0.01755079694742901,
                        -0.05265239084228703,
                        -0.07020318778971604,
                        -0.12285557863200308,
                        -0.1755079694742901,
                        -0.24571115726400616,
                        -0.3334651420011512,
                        -0.45632072063315426,
                        -0.6142778931600154,
                        -0.8073366595817345,
                        -1.0179462229508827,
                        -1.2812081771623178,
                        -1.579571725268611,
                        -1.930587664217191,
                        -2.2991544001132005,
                        -2.7203735268514966,
                        -3.141592653589793]

        Delta_list =   [0.,
                        0.13157895,
                        0.26315789,
                        0.39473684,
                        0.52631579,
                        0.65789474,
                        0.78947368,
                        0.92105263,
                        1.05263158,
                        1.18421053,
                        1.31578947,
                        1.44736842,
                        1.57894737,
                        1.71052632,
                        1.84210526,
                        1.97368421,
                        2.10526316,
                        2.23684211,
                        2.36842105,
                        2.5]

        def find_le(a, x):
            'Find rightmost value less than or equal to x'
            i = bisect.bisect_right(a, x)
            if i:
                return i
            raise ValueError
        z = find_le(np.abs(theta_list),abs(theta))

        x1 = Delta_list[z-1]
        x2 = Delta_list[z]
        y1 = theta_list[z-1]
        y2 = theta_list[z]
        y = theta
        x = (y-y1)*(x2-x1)/(y2-y1)+x1
        Delta0 = x
        # detuning
        def Delta(t,args):
            Delta0 = args['Delta0']
            return Delta0 * pow(np.sin(np.pi*t/t_total),2)

        tlist = np.linspace(0,t_total,t_total*10+1)
        dt_list = tlist[1:] - tlist[:-1]
        dt_list = np.append(dt_list,dt_list[0])
        self.dt_list[idx].append(dt_list)
        index = self.labels.index("\sigma^x_%d" % q)
        self.coeff_list[index].append(list(Delta(tlist, args = {'Delta0': Delta0})))

    def carb_dec(self,gate,idx):
        """
        Compiler for the CARB gate
        """
        targets = gate.targets # targets
        Theta = gate.arg_value % (2*np.pi) # argument
        t_total = 2 # total gate time

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
            return np.pi*Theta/(8*t_total*pow(self.alpha,2))*np.sin(np.pi*t/t_total)

        tlist = np.linspace(0,t_total,t_total*10+1)
        dt_list = tlist[1:] - tlist[:-1]
        dt_list = np.append(dt_list,dt_list[0])
        self.dt_list[idx].append(dt_list)

        index = self.labels.index("\sigma^z_%d\sigma^z_%d" % (targets[0], targets[1]))
        self.coeff_list[index].append(list(g(tlist, args = {'Theta': Theta})))
