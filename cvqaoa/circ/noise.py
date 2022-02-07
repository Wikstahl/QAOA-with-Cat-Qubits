from typing import Iterable
import numpy
import networkx
import cirq
import os

__all__ = ['CVZZChannel', 'CVRXChannel', 'DVRXChannel', 'DVZZChannel']


class Location():
    def __init__(self) -> None:
        # get path to directory
        os.chdir(os.path.dirname(__file__))
        self.loc = '../../data/kraus/'


class CVRXChannel(cirq.SingleQubitGate, Location):
    def __init__(self, arg: float) -> None:
        super().__init__()
        self.arg = arg % numpy.pi
        file = self.loc + 'cv_kraus_rx.npz'
        data = numpy.load(file, allow_pickle=True, fix_imports=True)
        self.kraus = data['kraus']
        self.arg_list = data['args']

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        # Angle of rotation
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = self.arg_list

        # Find elem in arg_list that arg is the closest too
        def absolute_difference_function(
            list_value): return abs(list_value - arg)
        closest_value = min(arg_list, key=absolute_difference_function)

        # Find the corresponding index
        idx = int(numpy.where(arg_list == closest_value)[0])

        # Get the kraus
        kraus = self.kraus[idx]
        return kraus

    def _has_mixture_(self) -> bool:
        return False

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, arg) -> str:
        return f"CX^{self.arg}"


class DVRXChannel(cirq.SingleQubitGate, Location):
    def __init__(self, arg: float) -> None:
        super().__init__()
        self.arg = arg % numpy.pi
        file = self.loc + 'dv_kraus_rx.npz'
        data = numpy.load(file, allow_pickle=True, fix_imports=True)
        self.kraus = data['kraus']
        self.arg_list = data['args']

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = self.arg_list

        # Find elem in arg_list that arg is the closest too
        def absolute_difference_function(
            list_value): return abs(list_value - arg)
        closest_value = min(arg_list, key=absolute_difference_function)

        # Find the corresponding index
        idx = int(numpy.where(arg_list == closest_value)[0])

        # Get the kraus
        kraus = self.kraus[idx]
        return kraus

    def _has_mixture_(self) -> bool:
        return False

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"DX^{self.arg}"


class DVZZChannel(cirq.Gate, Location):
    def __init__(self, arg) -> None:
        super().__init__()
        self.arg = arg % numpy.pi
        file = self.loc + 'dv_kraus_zz.npz'
        self.kraus = numpy.load(file, allow_pickle=True, fix_imports=True)
        data = numpy.load(file, allow_pickle=True, fix_imports=True)
        self.kraus = data['kraus']
        self.arg_list = data['args']

    def num_qubits(self) -> int:
        return 2

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = self.arg_list

        # Find elem in arg_list that arg is the closest too
        def absolute_difference_function(
            list_value): return abs(list_value - arg)
        closest_value = min(arg_list, key=absolute_difference_function)

        # Find the corresponding index
        idx = int(numpy.where(arg_list == closest_value)[0])

        # Get the kraus
        kraus = self.kraus[idx]
        return kraus

    def _has_mixture_(self) -> bool:
        return False

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(
            self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('DZZ', f"DZZ^{self.arg}"))

class CVZZChannel(cirq.Gate, Location):
    def __init__(self) -> None:
        super().__init__()
        file = self.loc + 'cv_kraus_zz.npy'
        self.kraus = numpy.load(file)

    def num_qubits(self) -> int:
        return 2

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        return list(self.kraus)

    def _has_mixture_(self) -> bool:
        return False

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(
            self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('CZZ', 'CZZ'))