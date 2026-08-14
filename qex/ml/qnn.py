import numpy as np
import cirq
from typing import Any, Tuple
from qex.ml import _require_ml

_require_ml("QNN")
import torch
import torch.nn as nn


class QuantumFunction(torch.autograd.Function):
    # PyTorch custom autograd function using Parameter Shift Rule.

    @staticmethod
    def forward(ctx: Any, weights: torch.Tensor, x: torch.Tensor, num_qubits: int, layers: int) -> torch.Tensor:
        ctx.save_for_backward(weights, x)
        ctx.num_qubits = num_qubits
        ctx.layers = layers

        weights_np = weights.detach().numpy()
        x_np = x.detach().numpy()
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

        simulator = cirq.Simulator()
        circuit = cirq.Circuit()

        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(float(x_np[i % len(x_np)]))(q))

        param_idx = 0
        for _ in range(layers):
            for i, q in enumerate(qubits):
                if param_idx < len(weights_np):
                    circuit.append(cirq.ry(weights_np[param_idx])(q))
                    param_idx += 1
                if param_idx < len(weights_np):
                    circuit.append(cirq.rz(weights_np[param_idx])(q))
                    param_idx += 1
            for i in range(num_qubits - 1):
                circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))

        res = simulator.simulate(circuit)
        state = res.final_state_vector
        z0 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        z_full = np.kron(z0, np.eye(2 ** (num_qubits - 1)))
        exp_val = float(np.real(np.vdot(state, z_full @ state)))

        return torch.tensor([exp_val], dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        weights, x = ctx.saved_tensors
        num_qubits = ctx.num_qubits
        layers = ctx.layers
        weights_np = weights.detach().numpy()
        num_params = len(weights_np)
        grads = np.zeros(num_params, dtype=np.float32)
        shift = np.pi / 2.0

        for i in range(num_params):
            w_plus = weights_np.copy()
            w_minus = weights_np.copy()
            w_plus[i] += shift
            w_minus[i] -= shift

            f_plus = QuantumFunction.apply(torch.tensor(w_plus, dtype=torch.float32), x, num_qubits, layers)
            f_minus = QuantumFunction.apply(torch.tensor(w_minus, dtype=torch.float32), x, num_qubits, layers)
            grads[i] = float((f_plus - f_minus) / 2.0)

        return grad_output * torch.tensor(grads, dtype=torch.float32), None, None, None


class QuantumLayer(nn.Module):
    # PyTorch neural layer wrapping a parameterized quantum circuit.

    def __init__(self, num_qubits: int = 2, layers: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.num_params = 2 * num_qubits * layers
        self.weights = nn.Parameter(torch.tensor(np.random.randn(self.num_params) * 0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumFunction.apply(self.weights, x, self.num_qubits, self.layers)


class QNN(nn.Module):
    # Hybrid Quantum Neural Network classifier module.

    def __init__(self, in_features: int = 2, num_qubits: int = 2):
        super().__init__()
        self.quantum_layer = QuantumLayer(num_qubits=num_qubits, layers=1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum_layer(x)
        return torch.sigmoid(self.fc(q_out))
