"""
Built-in demo experiments for validation.
"""

from typing import Dict, Any
import cirq
from qlab.experiment import Experiment


def x_gate_experiment() -> Experiment:
    """
    Demo: |0⟩ → X → |1⟩
    
    Simple X gate that flips |0⟩ to |1⟩.
    
    Returns:
        Experiment with no parameters.
    """
    def builder(qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit:
        """Build circuit: X gate on qubit."""
        return cirq.Circuit(cirq.X(qubit))
    
    return Experiment(name="x_gate", builder=builder)


def hadamard_experiment() -> Experiment:
    """
    Demo: |0⟩ → H → superposition
    
    Hadamard gate creating equal superposition |+⟩ = (|0⟩ + |1⟩)/√2.
    
    Returns:
        Experiment with no parameters.
    """
    def builder(qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit:
        """Build circuit: Hadamard gate on qubit."""
        return cirq.Circuit(cirq.H(qubit))
    
    return Experiment(name="hadamard", builder=builder)


def ry_sweep_experiment() -> Experiment:
    """
    Demo: |0⟩ → Ry(θ) sweep
    
    Rotation around Y-axis with parameter θ.
    This experiment expects params = {"theta": float}.
    
    Returns:
        Experiment that takes "theta" parameter (in radians).
    """
    def builder(qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit:
        """Build circuit: Ry(theta) rotation on qubit."""
        theta = params.get("theta", 0.0)
        if not isinstance(theta, (int, float)):
            raise ValueError(f"theta must be a number, got {type(theta)}")
        return cirq.Circuit(cirq.ry(theta)(qubit))
    
    return Experiment(name="ry_sweep", builder=builder)
