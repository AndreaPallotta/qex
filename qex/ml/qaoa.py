import numpy as np
import cirq
from typing import Dict, Any, List, Tuple
from qex.ml import _require_ml

_require_ml("QAOA")
import torch


class QAOA:
    # Quantum Approximate Optimization Algorithm (QAOA) for graph Max-Cut problems.

    def __init__(self, num_qubits: int = 4, p: int = 2):
        self.num_qubits = num_qubits
        self.p = p
        self.qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    def create_qaoa_circuit(self, gamma: np.ndarray, beta: np.ndarray, edges: List[Tuple[int, int]]) -> cirq.Circuit:
        # Constructs QAOA ansatz circuit with alternating cost and mixer Hamiltonians.
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*self.qubits))

        for step in range(self.p):
            g = float(gamma[step])
            b = float(beta[step])

            # Cost Hamiltonian phase separator.
            for u, v in edges:
                circuit.append(cirq.ZZ(self.qubits[u], self.qubits[v]) ** (g / np.pi))

            # Mixer Hamiltonian transverse field.
            for q in self.qubits:
                circuit.append(cirq.rx(2.0 * b)(q))

        return circuit

    def solve_maxcut(self, edges: List[Tuple[int, int]] = None, epochs: int = 30) -> Dict[str, Any]:
        # Solves Max-Cut graph partitioning problem on asymmetric graph.
        if edges is None:
            edges = [(0, 1), (0, 2), (0, 3), (1, 2)]

        num_params = 2 * self.p
        params_torch = torch.nn.Parameter(torch.tensor(np.random.rand(num_params) * 0.5, dtype=torch.float32))
        optimizer = torch.optim.Adam([params_torch], lr=0.08)

        loss_history = []
        simulator = cirq.Simulator()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            p_np = params_torch.detach().numpy()
            gamma, beta = p_np[: self.p], p_np[self.p :]

            circuit = self.create_qaoa_circuit(gamma, beta, edges)
            res = simulator.simulate(circuit)
            probs = np.abs(res.final_state_vector) ** 2

            # Compute Max-Cut expectation value.
            cut_expectation = 0.0
            for state_idx in range(2**self.num_qubits):
                bitstring = f"{state_idx:0{self.num_qubits}b}"
                cut_val = 0
                for u, v in edges:
                    if bitstring[u] != bitstring[v]:
                        cut_val += 1
                cut_expectation += probs[state_idx] * cut_val

            loss = torch.tensor(-cut_expectation, dtype=torch.float32, requires_grad=True)
            loss.backward()
            optimizer.step()

            loss_history.append(float(-loss.item()))

        final_circuit = self.create_qaoa_circuit(params_torch.detach().numpy()[: self.p], params_torch.detach().numpy()[self.p :], edges)
        final_res = simulator.simulate(final_circuit)

        return {
            "num_qubits": self.num_qubits,
            "max_cut_expectation": float(loss_history[-1]),
            "loss_history": loss_history,
            "state_vector": final_res.final_state_vector.tolist(),
        }
