"""
Quantum Fourier Transform (QFT) implementation in Cirq.
"""

from typing import List, Dict, Any
import numpy as np
import cirq
from qex.experiment import Experiment


class QFTExperiment(Experiment):
    """
    n-qubit Quantum Fourier Transform (QFT) circuit implementation.
    """

    def __init__(self, num_qubits: int = 3):
        self.num_qubits = num_qubits
        super().__init__(
            name=f"qft_{num_qubits}qubit",
            builder=self.build_circuit
        )

    def build_circuit(self, qubits: List[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit:
        circuit = cirq.Circuit()
        n = len(qubits)
        
        # Prepare initial state (e.g. |1> or superposition if specified)
        init_state = params.get("init_state", 1)
        for i in range(n):
            if (init_state >> (n - 1 - i)) & 1:
                circuit.append(cirq.X(qubits[i]))

        # Apply QFT gates
        for i in range(n):
            circuit.append(cirq.H(qubits[i]))
            for j in range(i + 1, n):
                k = j - i + 1
                phase = 2.0 / (2**k)
                circuit.append(cirq.CZ(qubits[j], qubits[i])**phase)

        # Swap qubits for standard ordering
        for i in range(n // 2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n - 1 - i]))

        return circuit
