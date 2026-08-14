"""
Grover's Quantum Search Algorithm implementation in Cirq.
"""

from typing import List, Dict, Any
import numpy as np
import cirq
from qex.experiment import Experiment


class GroverExperiment(Experiment):
    """
    3-qubit Grover's Search Algorithm searching for marked target state.
    """

    def __init__(self, target_state: int = 5):
        self.target_state = target_state
        super().__init__(
            name="grover_search",
            builder=self.build_circuit
        )

    def build_circuit(self, qubits: List[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        circuit = cirq.Circuit()
        target = params.get("target_state", self.target_state)
        n = min(3, len(qubits))
        q = qubits[:n]

        # Step 1: Equal Superposition
        for qubit in q:
            circuit.append(cirq.H(qubit))

        # Step 2: Oracle for target bitstring
        for bit_idx in range(n):
            if not ((target >> (n - 1 - bit_idx)) & 1):
                circuit.append(cirq.X(q[bit_idx]))

        # Multi-controlled Z flip
        circuit.append(cirq.H(q[n - 1]))
        circuit.append(cirq.TOFFOLI(q[0], q[1], q[2]))
        circuit.append(cirq.H(q[n - 1]))

        for bit_idx in range(n):
            if not ((target >> (n - 1 - bit_idx)) & 1):
                circuit.append(cirq.X(q[bit_idx]))

        # Step 3: Diffusion Operator (Amplifier)
        for qubit in q:
            circuit.append(cirq.H(qubit))
            circuit.append(cirq.X(qubit))

        circuit.append(cirq.H(q[n - 1]))
        circuit.append(cirq.TOFFOLI(q[0], q[1], q[2]))
        circuit.append(cirq.H(q[n - 1]))

        for qubit in q:
            circuit.append(cirq.X(qubit))
            circuit.append(cirq.H(qubit))

        return circuit
