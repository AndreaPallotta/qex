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
    For MVP, all backends must support 1-qubit circuits only.
    """
    
    @abstractmethod
    def run(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Execute a circuit and return the final density matrix.
        
        Args:
            circuit: A Cirq Circuit (must be 1-qubit for MVP).
            
        Returns:
            A 2x2 density matrix as a numpy array (complex dtype).
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
            circuit: A Cirq Circuit (must be 1-qubit for MVP).
            
        Returns:
            2x2 density matrix (complex dtype).
        """
        # Validate 1-qubit constraint
        qubits = circuit.all_qubits()
        if len(qubits) != 1:
            raise ValueError(f"Circuit must have exactly 1 qubit, got {len(qubits)}")
        
        # Run simulation to get final state
        result = self._simulator.simulate(circuit)
        statevector = result.final_state_vector
        
        # Convert statevector to density matrix: |ψ⟩⟨ψ|
        # For 1 qubit, statevector is length 2
        rho = np.outer(statevector, np.conj(statevector))
        
        return rho
    
    def get_name(self) -> str:
        """
        Get backend name.
        
        Returns:
            "cirq_ideal"
        """
        return "cirq_ideal"
