"""
Quantum Kernel & Support Vector Machine (QSVM) Classifier for SciML.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cirq


class QuantumKernel:
    """
    Computes quantum kernel matrix K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|² using Cirq.
    """

    def __init__(self, num_qubits: int = 2):
        self.num_qubits = num_qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]

    def feature_map(self, x: np.ndarray) -> cirq.Circuit:
        """Map feature vector x to quantum state |ψ(x)⟩."""
        circuit = cirq.Circuit()
        for i in range(self.num_qubits):
            circuit.append(cirq.H(self.qubits[i]))
            circuit.append(cirq.rz(2.0 * x[i % len(x)])(self.qubits[i]))
        if self.num_qubits >= 2:
            circuit.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
            circuit.append(cirq.rz(2.0 * (np.pi - x[0]) * (np.pi - x[1 % len(x)]))(self.qubits[1]))
            circuit.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
        return circuit

    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum fidelity overlap |⟨ψ(x1)|ψ(x2)⟩|²."""
        c1 = self.feature_map(x1)
        c2 = self.feature_map(x2)
        c_inv = cirq.inverse(c2)
        full_circuit = c1 + c_inv

        simulator = cirq.Simulator()
        result = simulator.simulate(full_circuit)
        state_vec = result.final_state_vector
        fidelity = float(np.abs(state_vec[0]) ** 2)
        return fidelity

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Compute N x N quantum Gram kernel matrix."""
        N = len(X)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.compute_kernel_element(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
        return K


class QSVMClassifier:
    """
    Quantum Support Vector Machine classifier with dual optimization.
    """

    def __init__(self, num_qubits: int = 2, C: float = 1.0):
        self.kernel = QuantumKernel(num_qubits=num_qubits)
        self.C = C
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit QSVM on training data X, y (y in {-1, +1})."""
        self.X_train = X
        self.y_train = y
        N = len(X)
        K = self.kernel.fit_transform(X)

        # Simple gradient ascent dual solver for alphas
        alpha = np.zeros(N)
        lr = 0.01
        for epoch in range(100):
            grad = np.ones(N) - (K * np.outer(y, y)) @ alpha
            alpha += lr * grad
            alpha = np.clip(alpha, 0, self.C)

        self.alpha = alpha
        # Bias calculation
        sv_idx = np.where(alpha > 1e-4)[0]
        if len(sv_idx) > 0:
            self.b = float(np.mean(y[sv_idx] - (K[sv_idx] * y) @ alpha))

        train_preds = np.sign(K @ (alpha * y) + self.b)
        accuracy = float(np.mean(train_preds == y))
        return {
            "kernel_matrix": K.tolist(),
            "accuracy": accuracy,
            "support_vectors": len(sv_idx)
        }
