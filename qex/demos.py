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


def quantum_teleportation_experiment() -> Experiment:
    """
    Demo: Quantum Teleportation protocol (3 qubits: Message, Alice, Bob).

    Teleports state |psi> from Qubit 0 to Qubit 2 using shared Bell pair (Qubits 1 and 2).

    Returns:
        Experiment for 3-qubit state teleportation.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if len(qubits) < 3:
            raise ValueError("Quantum Teleportation requires at least 3 qubits")
        q0, q1, q2 = qubits[0], qubits[1], qubits[2]

        # Prepare arbitrary message state |psi> = Ry(theta) |0>
        theta = params.get("theta", 1.2)
        circuit = cirq.Circuit(
            cirq.ry(theta)(q0),
            # Entangle Alice (q1) and Bob (q2)
            cirq.H(q1),
            cirq.CNOT(q1, q2),
            # Alice Bell-state measurement
            cirq.CNOT(q0, q1),
            cirq.H(q0),
            # Bob correction operations
            cirq.CNOT(q1, q2),
            cirq.CZ(q0, q2),
        )
        return circuit

    return Experiment(name="quantum_teleportation", builder=builder)


def ghz_state_experiment() -> Experiment:
    """
    Demo: Greenberger-Horne-Zeilinger (GHZ) state preparation (|000...0> + |111...1>) / sqrt(2).

    Returns:
        Experiment creating N-qubit GHZ state.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if not qubits:
            raise ValueError("GHZ state experiment requires at least 1 qubit")
        ops = [cirq.H(qubits[0])]
        for i in range(len(qubits) - 1):
            ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        return cirq.Circuit(ops)

    return Experiment(name="ghz_state", builder=builder)


def qaoa_maxcut_experiment() -> Experiment:
    """
    Demo: QAOA Max-Cut graph partitioning circuit.

    Returns:
        Experiment for QAOA Max-Cut on 4 qubits.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if len(qubits) < 4:
            qubits = [cirq.LineQubit(i) for i in range(4)]
        gamma = params.get("gamma", 0.5)
        beta = params.get("beta", 0.3)
        edges = [(0, 1), (0, 2), (0, 3), (1, 2)]
        c = cirq.Circuit(cirq.H.on_each(*qubits[:4]))
        for u, v in edges:
            c.append(cirq.ZZ(qubits[u], qubits[v]) ** (gamma / 3.14159))
        for q in qubits[:4]:
            c.append(cirq.rx(2.0 * beta)(q))
        return c

    return Experiment(name="qaoa_maxcut", builder=builder)


def qft_experiment() -> Experiment:
    """
    Demo: Quantum Fourier Transform (QFT) circuit on 3 qubits.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        from qex.algorithms.qft import QFTExperiment
        return QFTExperiment(num_qubits=min(4, max(2, len(qubits)))).build_circuit(qubits, params)

    return Experiment(name="qft_3qubit", builder=builder)


def grover_experiment() -> Experiment:
    """
    Demo: Grover's Search Algorithm searching for marked target state.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        from qex.algorithms.grover import GroverExperiment
        return GroverExperiment(target_state=params.get("target_state", 5)).build_circuit(qubits, params)

    return Experiment(name="grover_search", builder=builder)



def vqe_h2_experiment() -> Experiment:
    """
    Demo: VQE ansatz circuit for H2 molecule ground state optimization.

    Returns:
        Experiment for VQE H2 ansatz.
    """
    def builder(qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        if len(qubits) < 2:
            qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        theta = params.get("theta", 0.2)
        c = cirq.Circuit(
            cirq.X(qubits[0]),
            cirq.ry(theta)(qubits[0]),
            cirq.ry(theta)(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
        )
        return c

    return Experiment(name="vqe_h2", builder=builder)
