from typing import Iterable
import numpy
import networkx
import cirq

__all__ = ['CVRZChannel', 'CVRXChannel', 'DVRXChannel', 'DVZZChannel']


class Location():
    def __init__(self) -> None:
        self.loc = '../data/kraus/'


class CVRZChannel(cirq.SingleQubitGate, Location):
    def __init__(self) -> None:
        super().__init__()
        file = self.loc + 'cv_kraus_rz.npy'
        self.kraus = numpy.load(file, allow_pickle=True, fix_imports=True)

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        return self.kraus

    def _has_mixture_(self) -> bool:
        return False

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, arg) -> str:
        return f"DZ"


class CVRXChannel(cirq.SingleQubitGate, Location):
    def __init__(self, arg: float) -> None:
        super().__init__()
        self.arg = arg % numpy.pi
        file = self.loc + 'cv_kraus_rx.npy'
        self.kraus = numpy.load(file, allow_pickle=True, fix_imports=True)

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        # Angle of rotation
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = numpy.linspace(0, numpy.pi, num=181, endpoint=False)

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
        return f"DX^{self.arg}"


class DVRXChannel(cirq.SingleQubitGate, Location):
    def __init__(self, arg: float) -> None:
        super().__init__()
        file = self.loc + 'dv_kraus_rx.npy'
        self.kraus = numpy.load(file, allow_pickle=True, fix_imports=True)
        self.arg = arg % numpy.pi

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = numpy.linspace(0, numpy.pi, num=181, endpoint=False)

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
        file = self.loc + 'dv_kraus_zz.npy'
        self.kraus = numpy.load(file, allow_pickle=True, fix_imports=True)
        self.arg = arg % numpy.pi

    def num_qubits(self) -> int:
        return 2

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        arg = self.arg

        # This was the list used for creating the Kraus operators for
        # a given angle
        arg_list = numpy.linspace(0, numpy.pi, num=181, endpoint=False)

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
