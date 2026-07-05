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
    Demo: Create Bell state (for 2 qubits) or GHZ state (for N qubits).

    Applies H on first qubit, then CNOT sequentially to form entangled state.

    Returns:
        Experiment with no parameters (uses N qubits).
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if not qubits:
            raise ValueError("Bell/GHZ state experiment requires at least 1 qubit")
        operations = [cirq.H(qubits[0])]
        for i in range(len(qubits) - 1):
            operations.append(cirq.CNOT(qubits[i], qubits[i+1]))
        return cirq.Circuit(operations)

    return Experiment(name="bell_state", builder=builder)
