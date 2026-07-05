"""
Backend abstraction: interface for executing quantum circuits.
"""

from abc import ABC, abstractmethod
import cirq
import numpy as np


class Backend(ABC):
    """
    Abstract interface for executing quantum circuits.

    A backend executes a circuit and returns the final density matrix.
    Supports circuits with any number of qubits.
    """

    @abstractmethod
    def run(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Execute a circuit and return the final density matrix.

        Args:
            circuit: A Cirq Circuit (any number of qubits).

        Returns:
            Density matrix of shape (2**n, 2**n) for n qubits (complex dtype).
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name/identifier of this backend.

        Returns:
            Backend name string.
        """
        pass


class CirqBackend(Backend):
    """
    Concrete backend using Cirq's ideal simulator.

    Uses Cirq's Simulator for ideal (noiseless) simulation.
    Results are returned as density matrices derived from statevectors.
    """

    def __init__(self):
        """
        Initialize the Cirq ideal simulator backend.
        """
        self._simulator = cirq.Simulator()

    def run(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Execute circuit on ideal Cirq simulator and return density matrix.

        Args:
            circuit: A Cirq Circuit (any number of qubits).

        Returns:
            Density matrix (2**n x 2**n for n qubits, complex dtype).
        """
        result = self._simulator.simulate(circuit)
        statevector = result.final_state_vector
        # |ψ⟩⟨ψ| in same qubit order as Cirq (big-endian by default)
        rho = np.outer(statevector, np.conj(statevector))
        return rho

    def get_name(self) -> str:
        """
        Get backend name.

        Returns:
            "cirq_ideal"
        """
        return "cirq_ideal"


class NoisyCirqBackend(Backend):
    """
    Concrete backend using Cirq's density matrix simulator for noisy simulation.

    Simulates noisy circuits. Noise model can be configured; defaults to depolarizing noise.
    """

    def __init__(self, p: float = 0.0):
        """
        Initialize the noisy backend with depolarizing noise parameter.

        Args:
            p: Depolarizing noise probability. Must be in [0.0, 1.0].
        """
        self._simulator = cirq.DensityMatrixSimulator()
        self.p = p

    def run(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Execute circuit with depolarizing noise and return density matrix.

        Args:
            circuit: A Cirq Circuit (any number of qubits).

        Returns:
            Density matrix (2**n x 2**n for n qubits, complex dtype).
        """
        if self.p > 0.0:
            noisy_circuit = circuit.with_noise(cirq.depolarize(self.p))
        else:
            noisy_circuit = circuit

        result = self._simulator.simulate(noisy_circuit)
        return result.final_density_matrix

    def get_name(self) -> str:
        """
        Get backend name.

        Returns:
            "cirq_noisy_<p>"
        """
        return f"cirq_noisy_{self.p:.4f}"
