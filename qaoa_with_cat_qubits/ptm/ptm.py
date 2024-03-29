import numbers
import os
import bisect
from collections.abc import Iterable
from itertools import product, starmap, chain
from functools import partial, reduce
from operator import mul
import scipy as sp
import cirq

import numpy as np

pi = np.pi
from qutip import *

__all__ = [
    "PTM",
    "rho_to_pauli_basis",
    "pauli_basis_to_rho",
    "rz_ptm",
    "rx_ptm",
    "carb_ptm",
    "ptm_expand_1toN",
    "ptm_expand_2toN",
    "compute_pauli_transfer_matrix",
    "compute_process_fidelity",
]


def compute_pauli_transfer_matrix(circuit, num_qubits: int):
    # Get the Pauli matrices
    pauli_i = qeye(2)
    pauli_x = sigmax()
    pauli_y = sigmay()
    pauli_z = sigmaz()

    # Dimesion
    d = 2

    # List of Pauli matrices
    paulis = [pauli_i, pauli_x, pauli_y, pauli_z]

    # Initialize the Pauli transfer matrix
    ptm = np.zeros((4, 4), dtype=complex)

    # Initialize density matrix simulator
    simulator = cirq.DensityMatrixSimulator(split_untangled_states=True)

    # Initialize qubits
    qubits = cirq.LineQubit.range(num_qubits)

    # Apply the quantum channel / circuit to each Pauli operator
    for i in range(len(paulis)):
        eVals, eVecs = sp.linalg.eigh(paulis[i].full())

        # list of density matrices
        density_matrices = []

        # Since we only have two eigenvalues
        for j in range(len(eVals)):
            # Prepare initial state |P_i 0...0>
            initial_state = tensor(
                Qobj(eVecs[:, j]), tensor([basis(2, 0)] * (num_qubits - 1))
            )
            initial_state = initial_state.full().flatten()

            # Simulate the circuit
            result = simulator.simulate(
                circuit,
                initial_state=initial_state,
                qubit_order=cirq.LineQubit.range(num_qubits),
            )

            # Final state density matrix
            rho = Qobj(
                result.final_density_matrix, dims=[[2] * num_qubits, [2] * num_qubits]
            )

            # Partial trace
            rho = rho.ptrace(0)

            # Append rho to list of density marices
            density_matrices.append(rho)

        for j in range(len(paulis)):
            # Calculate the inner product (trace) of the final state with each Pauli matrix
            coefficient = (
                eVals[0] * (paulis[j] * density_matrices[0]).tr()
                + eVals[1] * (paulis[j] * density_matrices[1]).tr()
            ) / d

            ptm[j, i] = coefficient

    return ptm


def compute_process_fidelity(circuit_a, circuit_b, num_qubits: int):
    # Get the Pauli matrices
    pauli_i = qeye(2)
    pauli_x = sigmax()
    pauli_y = sigmay()
    pauli_z = sigmaz()

    # Dimesion
    d = 2

    # List of Pauli matrices
    paulis = [pauli_i, pauli_x, pauli_y, pauli_z]

    # Initialize the process fidelity
    process_fidelity = []

    # Initialize density matrix simulator
    simulator = cirq.DensityMatrixSimulator(split_untangled_states=True)

    # Initialize qubits
    qubits = cirq.LineQubit.range(num_qubits)

    # Apply the quantum channel / circuit to each Pauli operator
    for i in range(len(paulis)):
        eVals, eVecs = sp.linalg.eigh(paulis[i].full())

        # list of density matrices
        dm_a = []
        dm_b = []

        # Since we only have two eigenvalues
        for j in range(len(eVals)):
            # Prepare initial state |P_i 0...0>
            initial_state = tensor(
                Qobj(eVecs[:, j]), tensor([basis(2, 0)] * (num_qubits - 1))
            )
            initial_state = initial_state.full().flatten()

            # Simulate the circuit
            result_a = simulator.simulate(
                circuit_a,
                initial_state=initial_state,
                qubit_order=cirq.LineQubit.range(num_qubits),
            )

            result_b = simulator.simulate(
                circuit_b,
                initial_state=initial_state,
                qubit_order=cirq.LineQubit.range(num_qubits),
            )

            # Final state density matrix
            rho_a = Qobj(
                result_a.final_density_matrix, dims=[[2] * num_qubits, [2] * num_qubits]
            )
            rho_b = Qobj(
                result_b.final_density_matrix, dims=[[2] * num_qubits, [2] * num_qubits]
            )

            # Partial trace
            rho_a = rho_a.ptrace(0)
            rho_b = rho_b.ptrace(0)

            # Append rho to list of density marices
            dm_a.append(rho_a)
            dm_b.append(rho_b)

        # Calculate
        coefficient = (
            (eVals[0] * dm_a[0] + eVals[1] * dm_a[1])
            * (eVals[0] * dm_b[0] + eVals[1] * dm_b[1])
        ).tr()
        process_fidelity.append(coefficient)

    return process_fidelity


def rho_to_pauli_basis(rho):
    """
    Given a quantum operator write it in
    vector form in the Pauli basis.
    """
    dim = rho.shape[0]
    nq = int(np.log2(dim))
    dims = [[2] * nq, [1] * nq]
    data = np.zeros((dim**2, 1), dtype=np.complex64)
    pauli_basis = (
        qeye(2) / np.sqrt(2),
        sigmax() / np.sqrt(2),
        sigmay() / np.sqrt(2),
        sigmaz() / np.sqrt(2),
    )
    for idx, op in enumerate(starmap(tensor, product(pauli_basis, repeat=nq))):
        data[idx] = (op * rho).tr()
    return Qobj(data, dims=dims)


def pauli_basis_to_rho(rho):
    """
    Given a quantum operator in vector from that is written in the Pauli basis
    convert it to operator form in the computational basis.
    """
    dim = rho.shape[0]
    nq = int(np.log2(np.sqrt(dim)))
    dims = [[2] * nq, [2] * nq]
    data = rho.data.toarray()
    rho = 0
    pauli_basis = (
        qeye(2) / np.sqrt(2),
        sigmax() / np.sqrt(2),
        sigmay() / np.sqrt(2),
        sigmaz() / np.sqrt(2),
    )
    for idx, op in enumerate(starmap(tensor, product(pauli_basis, repeat=nq))):
        rho += op * complex(data[idx])
    return Qobj(rho, dims=dims)


def rz_ptm(arg_value):
    """
    The RZ-gate written in Pauli form
    """
    d = 2
    pauli_basis = (qeye(2), sigmax(), sigmay(), sigmaz())
    rz = (-1j * arg_value / 2 * sigmaz()).expm()
    R = np.zeros((d**2, d**2))
    for i in range(d**2):
        for j in range(d**2):
            R[i, j] = 1 / d * (pauli_basis[i] * rz * pauli_basis[j] * rz.dag()).tr()
    return Qobj(R, dims=[[2], [2]])


def rx_ptm(arg_value):
    """
    The RX-gate written in Pauli form
    """
    d = 2
    pauli_basis = (qeye(2), sigmax(), sigmay(), sigmaz())
    rx = (-1j * arg_value / 2 * sigmax()).expm()
    R = np.zeros((d**2, d**2))
    for i in range(d**2):
        for j in range(d**2):
            R[i, j] = 1 / d * (pauli_basis[i] * rx * pauli_basis[j] * rx.dag()).tr()
    return Qobj(R, dims=[[2], [2]])


def carb_ptm(arg_value):
    """
    The CARB-gate written in Pauli form
    """
    d = 4
    pauli_basis = (qeye(2), sigmax(), sigmay(), sigmaz())
    pauli_basis = list(starmap(tensor, product(pauli_basis, repeat=2)))
    carb = (-1j * arg_value / 2 * tensor(sigmaz(), sigmaz())).expm()
    R = np.zeros((d**2, d**2))
    for i in range(d**2):
        for j in range(d**2):
            R[i, j] = 1 / d * (pauli_basis[i] * carb * pauli_basis[j] * carb.dag()).tr()
    return Qobj(R, dims=[[2, 2], [2, 2]])


def ptm_expand_1toN(U, N, target):
    """
    Create a Qobj representing a one-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The one-qubit gate

    N : integer
        The number of qubits in the target space.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")
    temp = tensor([identity(4)] * (target) + [U] + [identity(4)] * (N - target - 1))
    temp.dims = [[2] * N, [2] * N]
    return temp


def ptm_expand_2toN(U, N, targets=None):
    """
    Create a Qobj representing a two-qubit PTM that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The two-qubit PTM

    N : integer
        The number of qubits in the target space.

    targets : list
        List of target qubits.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if targets is not None:
        control, target = targets

    if control is None or target is None:
        raise ValueError("Specify value of control and target")

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if control >= N or target >= N:
        raise ValueError("control and not target must be integer < integer N")

    if control == target:
        raise ValueError("target and not control cannot be equal")

    p = list(range(N))

    if target == 0 and control == 1:
        p[control], p[target] = p[target], p[control]

    elif target == 0:
        p[1], p[target] = p[target], p[1]
        p[1], p[control] = p[control], p[1]

    else:
        p[1], p[target] = p[target], p[1]
        p[0], p[control] = p[control], p[0]
    U.dims = [[4, 4], [4, 4]]
    temp = tensor([U] + [qeye(4)] * (N - 2)).permute(p)
    temp.dims = [[2] * N, [2] * N]
    return temp


class PTM(object):
    """docstring for ."""

    def __init__(self, num_lvl=20):
        self.num_lvl = num_lvl  # number of levels
        # annihilation operators
        a = destroy(num_lvl)
        eye = qeye(num_lvl)
        a1 = tensor([a, eye])
        a2 = tensor([eye, a])
        K = 1  # kerr amplitude
        G = 4 * K  # two photon pump amplitude
        self.alpha = np.sqrt(G / K)  # coherent state amplitude
        self.kappa = 1 / 1500  # single-photon loss rate

        # Single and two KNR collapse operators
        self.c_op = np.sqrt(self.kappa) * a
        self.c_ops = [np.sqrt(self.kappa) * a1, np.sqrt(self.kappa) * a2]

        # cat states
        cat_plus = (
            coherent(num_lvl, self.alpha) + coherent(num_lvl, -self.alpha)
        ).unit()
        cat_minus = (
            coherent(num_lvl, self.alpha) - coherent(num_lvl, -self.alpha)
        ).unit()

        # computational basis
        up = (cat_plus + cat_minus) / np.sqrt(2)
        down = (cat_plus - cat_minus) / np.sqrt(2)

        # Identity in computational basis
        I = up * up.dag() + down * down.dag()
        # sigma-z in computational basis
        sigma_z = up * up.dag() - down * down.dag()
        # sigma-x in computational basis
        sigma_x = up * down.dag() + down * up.dag()
        # sigma-y in computational basis
        sigma_y = 1j * (-up * down.dag() + down * up.dag())

        # Array with Pauli matrices
        P = [I, sigma_x, sigma_y, sigma_z]
        Q = []
        for i in range(4):
            Q.extend([tensor(P[i], P[j]) for j in range(4)])
        self.P = {"Single": P, "Double": Q}

        def E(t, args):
            arg_value = args["arg_value"]
            T_g = args["T_g"]
            return np.pi * arg_value / (8 * T_g * self.alpha) * np.sin(np.pi * t / T_g)

        # detuning
        def Delta(t, args):
            delta = args["delta"]
            T_g = args["T_g"]
            return delta * pow(np.sin(np.pi * t / T_g), 2)

        # Hamiltonian
        H0 = -K * pow(a.dag(), 2) * pow(a, 2) + G * (pow(a.dag(), 2) + pow(a, 2))
        self.H_I = [H0]
        self.H_RZ = [H0, [a.dag() + a, E]]
        self.H_RX = [H0, [-a.dag() * a, Delta]]

        # coupling
        def g(t, args):
            arg_value = args["rg_value"]
            T_g = args["T_g"]
            return (
                np.pi
                * arg_value
                / (8 * T_g * pow(self.alpha, 2))
                * np.sin(np.pi * t / T_g)
            )

        H1 = -K * pow(a1.dag(), 2) * pow(a1, 2) + G * (pow(a1.dag(), 2) + pow(a1, 2))
        H2 = -K * pow(a2.dag(), 2) * pow(a2, 2) + G * (pow(a2.dag(), 2) + pow(a2, 2))
        H_coupling = a1.dag() * a2 + a2.dag() * a1
        self.H_U = [(H1 + H2), [H_coupling, g]]  # Hamiltonian for the CARB-gate

    def pauli_transfer_matrix(self, H_tot, d, T_g, args=None, opt=None):
        """
        Return the Pauli transfer matrix given a Hamiltonian

        Parameters
        ----------
        H_tot: list of Qobj
            The total Hamiltonian of the system

        d: int
            Dimension of the gate
            (d=2 for single qubit gates and d=4 for two-qubit gates)

        T_g: float
            Total gate time

        args: dict
            Argument that is passed to the solver

        Returns
        -------
        Pauli transfer matrix: Qobj
        """
        if d == 2:
            P = self.P["Single"]
            c_ops = self.c_op
        elif d == 4:
            P = self.P["Double"]
            c_ops = self.c_ops
        else:
            TypeError("Something went wrong")
        # pauli transfer matrix
        R = np.zeros((d**2, d**2))
        R[0, 0] = 1
        for j in range(d**2):
            rho = mesolve(
                H_tot,
                P[j],
                [0, T_g],
                c_ops=c_ops,
                args=args,
                options=opt,
                progress_bar=True,
            )
            Lambda = rho.states[-1]
            for i in range(1, d**2):
                R[i, j] = 1 / d * np.real((P[i] * Lambda).tr())
        R = Qobj(R, dims=[[2] * int(d / 2), [2] * int(d / 2)])
        return R

    def id(self, arg_value):
        """
        Return the Pauli transfer matrix of the identity

        Parameters
        ----------

        arg_value: float
            time

        Returns
        -------
        Pauli transfer matrix: Qobj
        """
        T_g = arg_value
        d = 2
        H_tot = self.H_I
        opt = Options(nsteps=1e5)  # For precise calculation
        return self.pauli_transfer_matrix(H_tot, d, T_g, opt=opt)

    def rz(self, arg_value):
        """
        Return the Pauli transfer matrix of the RZ-gate

        Parameters
        ----------

        arg_value: float
            Rotational angle

        Returns
        -------
        Pauli transfer matrix: Qobj
        """
        T_g = 2
        d = 2
        H_tot = self.H_RZ
        args = {"arg_value": arg_value, "T_g": T_g}
        opt = Options(nsteps=1e4)  # For precise calculation
        return self.pauli_transfer_matrix(H_tot, d, T_g, args, opt)

    def rx(self, arg_value):
        """
        Return the Pauli transfer matrix of the RX-gate

        Parameters
        ----------

        arg_value: float
            Rotational angle

        Returns
        -------
        Pauli transfer matrix: Qobj
        """
        T_g = 10  # gate time
        d = 2
        theta = arg_value % np.pi
        theta_list = np.load("results/theta-list.npy")
        delta_list = np.load("results/delta-list.npy")

        def find_le(a, x):
            "Find rightmost value less than or equal to x"
            i = bisect.bisect_right(a, x)
            if i:
                return i
            raise ValueError

        # find which delta that corresponds to a specific theta
        z = find_le(theta_list, theta)
        (x1, x2) = (delta_list[z - 1], delta_list[z])
        (y1, y2) = (theta_list[z - 1], theta_list[z])
        y = theta
        x = (y - y1) * (x2 - x1) / (y2 - y1) + x1
        delta = x

        H_tot = self.H_RX
        args = {"delta": delta, "T_g": T_g}
        opt = Options(nsteps=2e4)  # For precise calculation
        return self.pauli_transfer_matrix(H_tot, d, T_g, args, opt)

    def carb(self, arg_value):
        """
        Return the Pauli transfer matrix of the U-gate

        Parameters
        ----------

        arg_value: float
            Rotational angle

        Returns
        -------
        Pauli transfer matrix: Qobj
        """
        T_g = 2
        d = 4
        H_tot = self.H_U
        args = {"arg_value": arg_value, "T_g": T_g}
        opt = Options(nsteps=1e4)  # For precise calculation
        return self.pauli_transfer_matrix(H_tot, d, T_g, args, opt)
