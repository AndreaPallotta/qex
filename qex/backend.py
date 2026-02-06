"""
Backend abstraction: interface for executing quantum circuits.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
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
