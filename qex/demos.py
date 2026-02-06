"""
Built-in demo experiments for validation.
"""

from typing import Dict, Any, Sequence
import cirq
from qex.experiment import Experiment


def x_gate_experiment() -> Experiment:
    """
    Demo: |0⟩ → X → |1⟩

    Simple X gate that flips |0⟩ to |1⟩ (acts on first qubit).

    Returns:
        Experiment with no parameters.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        return cirq.Circuit(cirq.X(qubits[0]))

    return Experiment(name="x_gate", builder=builder)


def hadamard_experiment() -> Experiment:
    """
    Demo: |0⟩ → H → superposition

    Hadamard gate creating equal superposition |+⟩ (on first qubit).

    Returns:
        Experiment with no parameters.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        return cirq.Circuit(cirq.H(qubits[0]))

    return Experiment(name="hadamard", builder=builder)


def ry_sweep_experiment() -> Experiment:
    """
    Demo: |0⟩ → Ry(θ) sweep

    Rotation around Y-axis with parameter θ (on first qubit).
    Expects params = {"theta": float} in radians.

    Returns:
        Experiment that takes "theta" parameter.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        theta = params.get("theta", 0.0)
        if not isinstance(theta, (int, float)):
            raise ValueError(f"theta must be a number, got {type(theta)}")
        return cirq.Circuit(cirq.ry(theta)(qubits[0]))

    return Experiment(name="ry_sweep", builder=builder)


def bell_state_experiment() -> Experiment:
    """
    Demo: Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

    Applies H on first qubit then CNOT(control=qubits[0], target=qubits[1]).

    Returns:
        Experiment with no parameters (uses 2 qubits).
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if len(qubits) < 2:
            raise ValueError("Bell state experiment requires at least 2 qubits")
        return cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
        )

    return Experiment(name="bell_state", builder=builder)
