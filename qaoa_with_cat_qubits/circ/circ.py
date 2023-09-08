from typing import Iterable
from itertools import product
import networkx
import numpy
import cirq
import os
from qaoa_with_cat_qubits.circ.noise import *
from qutip import tensor, qeye, sigmax, sigmay, sigmaz

__all__ = ['Circ', 'average_gate_fidelity']

def average_gate_fidelity(circuit:cirq.Circuit, U:numpy.ndarray) -> float:

    # List with the qubits
    qubits = list(circuit.all_qubits())
    # Number of qubits
    num_q = len(qubits)
    # Dimension
    d = 2**num_q

    # All Pauli matrices including the identity
    sigma_list = [qeye(2), sigmax(), sigmay(), sigmaz()]

    # List with all Pauli matrices
    sigma = map(tensor, map(list, product(sigma_list, repeat=num_q)))

    # Simulator
    sim = cirq.DensityMatrixSimulator(dtype = numpy.complex128)

    # Average Gate Fidelity
    F = 0
    for sigma_k in sigma:
        sigma_k = numpy.array(sigma_k.full())
        # Get eigenvalues and eigenvectors
        eVal, eVec = numpy.linalg.eigh(sigma_k)
        # Quantum map
        final_state = 0
        for w, v in zip(eVal, eVec.T):
            v = numpy.reshape(v,(d,1)) # reshape to a column vector
            input_state = v@numpy.conj(v.T) # input density matrix
            # Simulate circuit
            result = sim.simulate(
                circuit,
                initial_state=input_state,
                qubit_order=qubits
            )
            final_state += w*result.final_density_matrix
        # Target state
        target_state = U@sigma_k@numpy.conj(U.T)
        F += numpy.trace(target_state@final_state).real
    return (F + d**2) / (d**2*(d + 1))


class Circ(object):
    def __init__(self, graph: networkx.Graph) -> None:
        """Init

        Args:
            graph (networkx.Graph): A Max-Cut graph
        """
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.num_edges = len(graph.edges)
        self.cost = self.get_cost()

    def get_cost(self) -> numpy.ndarray:
        """
        The MaxCut cost values of a graph

        Returns:
            numpy.ndarray: The cost values as an 1D-array
        """
        def product(*args, repeat=1):
            # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
            # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
            pools = [list(pool) for pool in args] * repeat
            result = [[]]
            for pool in pools:
                result = [x + [y] for x in result for y in pool]
            for prod in result:
                yield list(prod)

        # Number of edges
        M = self.num_edges
        # Number of nodes
        N = self.num_nodes
        # Adjacency matrix
        A = networkx.adjacency_matrix(self.graph).todense()

        # Generate a list of all possible n‐tuples of elements from {1,-1} and
        # organize them as a (2^n x n) matrix. In other words create all
        # possible solutions to the problem.
        s = numpy.array(list(product([1, -1], repeat=N)))

        # Construct the the cost function for Max Cut: C=1/2*Sum(Z_i*Z_j)-M/2
        # Note: This is the minimization version
        return 1 / 2 * (numpy.diag(s@numpy.triu(A)@s.T) - M)

    def qaoa_circuit(self,
                     params: tuple,
                     device: str,
                     amplitude,
                     cutoff
                     ) -> cirq.Circuit:
        """Creates the first iteration of the QAOA circuit

        Args:

        Returns:
            cirq.Circuit: QAOA circuit
        """

        # The rotation angles in the QAOA circuit.
        alphas, betas = params

        qubits = cirq.LineQubit.range(self.num_nodes)  # Create qubits

        circuit = cirq.Circuit()  # Initialize circuit
        circuit.append(cirq.H(q) for q in qubits)  # Add Hadamard

        for alpha, beta in zip(alphas, betas):
            for (u, v) in self.graph.edges:
                circuit.append(cirq.ops.ZZPowGate(
                    exponent=(alpha / numpy.pi),
                    global_shift=-.5)(qubits[u], qubits[v])
                )
                if device == "DV":
                    circuit.append(DVZZChannel(alpha,amplitude,cutoff)(qubits[u], qubits[v]))
                if device == "CV":
                    circuit.append(CVZZChannel(amplitude,cutoff)(qubits[u], qubits[v]))

            circuit.append(
                cirq.Moment(
                    # This gate is equivalent to the RX-gate
                    # That is why we multiply by two in the exponent
                    cirq.ops.XPowGate(
                        exponent=(2 * beta / numpy.pi),
                        global_shift=-.5)(q) for q in qubits
                )
            )
            if device == "DV":
                circuit.append(DVRXChannel(beta,amplitude,cutoff).on_each(*qubits))
            if device == "CV":
                circuit.append(CVRXChannel(beta,amplitude,cutoff).on_each(*qubits))

        return circuit

    def simulate_qaoa(self,
                      params: tuple,
                      device: str,
                      amplitude,
                      cutoff
                      ) -> numpy.ndarray:
        """Simulates the p=1 QAOA circuit of a graph

        Args:
            params (tuple): Variational parameters
            device (str): CV or DV device

        Returns:
            numpy.ndarray: Density matrix output
        """
        alpha, beta = params
        circuit = self.qaoa_circuit(params, device, amplitude, cutoff)

        # prepare initial state |00...0>
        initial_state = numpy.zeros(2**self.num_nodes)
        initial_state[0] = 1

        # Density matrix simulator
        sim = cirq.DensityMatrixSimulator(
            split_untangled_states=True
        )

        # Simulate the QAOA
        result = sim.simulate(
            circuit,
            initial_state=initial_state,
            qubit_order=cirq.LineQubit.range(self.num_nodes)
        )
        return result.final_density_matrix

    def optimize_brute_qaoa(self, x: tuple, *args: tuple) -> float:
        """Optimization function for QAOA using brute force in Scipy optimize.

        Args:
            x (tuple): Variational parameters
            *args (tuple): Error Channel
        Returns:
            float: Expectation value
        """
        middle_index = int(len(x) / 2)
        alphas = tuple(x[:middle_index])
        betas = tuple(x[middle_index:])
        device = args[0][0]
        amplitude = args[0][1]
        cutoff = args[0][2]
        rho = self.simulate_qaoa(
            params=(alphas, betas),
            device=device,
            amplitude=amplitude,
            cutoff=cutoff
        )
        return numpy.trace(self.cost * rho).real

    def optimize_qaoa(self, x: numpy.ndarray, *args: tuple) -> float:
        """Optimization function for QAOA that is compatible with
            Scipy optimize.

        Args:
            x (numpy.ndarray): Variational parameters
            *args (tuple): Error Channel
        Returns:
            float: Expectation value
        """
        middle_index = int(len(x) / 2)
        alphas = tuple(x[:middle_index])
        betas = tuple(x[middle_index:])
        if len(args[0]) == 1:
            device = args[0][0]
            amplitude = 0 
            cutoff = 0
        else:
            device = args[0][0]
            amplitude = args[0][1]
            cutoff = args[0][2]
        if type(device) is list:
            device = device[0]
        rho = self.simulate_qaoa(
            params=(alphas, betas),
            device=device,
            amplitude=amplitude,
            cutoff=cutoff
        )
        return numpy.trace(self.cost * rho).real
