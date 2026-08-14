import numpy as np
import cirq
from typing import Dict, Any, List, Tuple
from qex.ml import _require_ml

_require_ml("VQE")
import torch


class VQE:
    # Variational Quantum Eigensolver for molecular and Pauli Hamiltonians.

    def __init__(self, num_qubits: int = 2, ansatz_type: str = "hardware_efficient", layers: int = 2):
        self.num_qubits = num_qubits
        self.ansatz_type = ansatz_type
        self.layers = layers
        self.qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    def create_ansatz(self, params_np: np.ndarray) -> cirq.Circuit:
        # Constructs parameterized Cirq ansatz circuit from parameter vector.
        circuit = cirq.Circuit()
        param_idx = 0

        for layer in range(self.layers):
            for i, q in enumerate(self.qubits):
                if param_idx < len(params_np):
                    circuit.append(cirq.ry(params_np[param_idx])(q))
                    param_idx += 1
                if param_idx < len(params_np):
                    circuit.append(cirq.rz(params_np[param_idx])(q))
                    param_idx += 1

            for i in range(self.num_qubits - 1):
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))

        return circuit

    def compute_expectation(self, circuit: cirq.Circuit, hamiltonian_terms: List[Tuple[float, List[Tuple[int, str]]]]) -> float:
        # Computes expectation value <psi|H|psi> across Pauli terms.
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state_vec = result.final_state_vector

        # Construct full Hamiltonian matrix.
        dim = 2**self.num_qubits
        h_matrix = np.zeros((dim, dim), dtype=np.complex128)

        pauli_dict = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

        for coeff, term in hamiltonian_terms:
            term_mat = np.eye(1)
            op_map = {q_idx: op for q_idx, op in term}
            for q in range(self.num_qubits):
                op_str = op_map.get(q, "I")
                term_mat = np.kron(term_mat, pauli_dict[op_str])
            h_matrix += coeff * term_mat

        expectation = float(np.real(np.vdot(state_vec, h_matrix @ state_vec)))
        return expectation

    def solve_h2(self, epochs: int = 40, lr: float = 0.05) -> Dict[str, Any]:
        # Solves H2 hydrogen molecule ground state energy at r=0.74A (-1.137 Ha).
        h2_hamiltonian = [
            (-1.05237, []),
            (0.39793, [(0, "Z")]),
            (-0.39793, [(1, "Z")]),
            (-0.01128, [(0, "Z"), (1, "Z")]),
            (0.18093, [(0, "X"), (1, "X")]),
        ]

        num_params = 2 * self.num_qubits * self.layers
        params_torch = torch.nn.Parameter(torch.tensor(np.random.randn(num_params) * 0.1, dtype=torch.float32))
        optimizer = torch.optim.Adam([params_torch], lr=lr)

        energy_history = []
        param_history = []

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            p_np = params_torch.detach().numpy()
            circuit = self.create_ansatz(p_np)
            energy = self.compute_expectation(circuit, h2_hamiltonian)

            # Parameter shift rule gradient estimation for autograd integration.
            grads = np.zeros(num_params, dtype=np.float32)
            shift = np.pi / 2.0
            for i in range(num_params):
                p_plus = p_np.copy()
                p_minus = p_np.copy()
                p_plus[i] += shift
                p_minus[i] -= shift

                c_plus = self.create_ansatz(p_plus)
                c_minus = self.create_ansatz(p_minus)

                e_plus = self.compute_expectation(c_plus, h2_hamiltonian)
                e_minus = self.compute_expectation(c_minus, h2_hamiltonian)

                grads[i] = (e_plus - e_minus) / (2.0 * np.sin(shift))

            params_torch.grad = torch.tensor(grads, dtype=torch.float32)
            optimizer.step()

            energy_history.append(float(energy))
            param_history.append(params_torch.detach().numpy().tolist())

        final_circuit = self.create_ansatz(params_torch.detach().numpy())
        final_sim = cirq.Simulator().simulate(final_circuit)

        return {
            "num_qubits": self.num_qubits,
            "final_energy": float(energy_history[-1]),
            "exact_h2_energy": -1.1373,
            "energy_history": energy_history,
            "param_history": param_history,
            "state_vector": final_sim.final_state_vector.tolist(),
        }
