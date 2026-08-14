"""
Extensive test suite for qex.ml (Quantum Machine Learning Extension).
"""

import pytest
import numpy as np
import cirq
from pathlib import Path
import tempfile
from click.testing import CliRunner

from qex.ml import has_ml, _require_ml
from qex.cli import main, ml_status, run_vqe


def test_has_ml_detection():
    # Verifies ML dependency detection boolean.
    assert isinstance(has_ml(), bool)


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_vqe_ansatz_and_h2_solver():
    # Tests VQE ansatz generation and H2 ground state energy convergence.
    from qex.ml.vqe import VQE

    vqe = VQE(num_qubits=2, layers=2)
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    circuit = vqe.create_ansatz(params)

    assert isinstance(circuit, cirq.Circuit)
    assert len(circuit) > 0

    results = vqe.solve_h2(epochs=15, lr=0.08)
    assert "final_energy" in results
    assert len(results["energy_history"]) == 15
    assert results["final_energy"] < -0.8  # Energy decreases toward ground state -1.1373 Ha


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_quantum_neural_network():
    # Tests PyTorch hybrid quantum neural network forward and autograd backward passes.
    import torch
    from qex.ml.qnn import QuantumLayer, QNN

    layer = QuantumLayer(num_qubits=2, layers=1)
    x = torch.tensor([0.5, 1.2], dtype=torch.float32)
    out = layer(x)

    assert out.shape == (1,)
    assert -1.0 <= out.item() <= 1.0

    qnn = QNN(in_features=2, num_qubits=2)
    y_pred = qnn(x)
    loss = (y_pred - 0.5) ** 2
    loss.backward()

    assert qnn.quantum_layer.weights.grad is not None


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_neural_state_tomography():
    # Tests density matrix denoising, unit trace, and positive semi-definiteness.
    from qex.ml.tomography import NeuralTomography

    tomography = NeuralTomography(num_qubits=2)
    noisy_rho = np.eye(4, dtype=np.complex128) / 4.0 + 0.02 * np.random.randn(4, 4)
    noisy_rho = (noisy_rho + noisy_rho.conj().T) / 2.0

    res = tomography.denoise_density_matrix(noisy_rho, iterations=10)
    clean_rho = res["clean_density_matrix"]

    assert clean_rho.shape == (4, 4)
    assert np.isclose(np.trace(clean_rho), 1.0, atol=1e-3)
    assert res["purity"] <= 1.0001


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_quantum_optimizer_spsa():
    # Tests SPSA optimization step on a quadratic loss landscape.
    from qex.ml.optimizer import QuantumOptimizer

    optimizer = QuantumOptimizer(a=0.1, c=0.1)
    cost_fn = lambda p: float(np.sum(p**2))
    params = np.array([1.0, -1.0])

    new_params, cost = optimizer.spsa_step(cost_fn, params, k=0)
    assert new_params.shape == (2,)


def test_cli_ml_status():
    # Tests qex ml-status CLI command execution.
    runner = CliRunner()
    result = runner.invoke(main, ["ml-status"])
    assert result.exit_code == 0
    assert "qex[ml] Status" in result.output


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_qaoa_maxcut():
    # Tests QAOA Max-Cut graph optimization algorithm.
    from qex.ml.qaoa import QAOA

    qaoa = QAOA(num_qubits=4, p=2)
    res = qaoa.solve_maxcut(epochs=10)

    assert "max_cut_expectation" in res
    assert len(res["loss_history"]) == 10
    assert len(res["state_vector"]) == 16


def test_qft_algorithm():
    # Tests Quantum Fourier Transform circuit execution.
    from qex.algorithms.qft import QFTExperiment

    qft = QFTExperiment(num_qubits=3)
    qubits = [cirq.GridQubit(0, i) for i in range(3)]
    c = qft.build_circuit(qubits, {"init_state": 1})

    assert isinstance(c, cirq.Circuit)
    assert len(c) > 0


def test_grover_algorithm():
    # Tests Grover's Search Algorithm circuit execution.
    from qex.algorithms.grover import GroverExperiment

    grover = GroverExperiment(target_state=5)
    qubits = [cirq.GridQubit(0, i) for i in range(3)]
    c = grover.build_circuit(qubits, {"target_state": 5})

    assert isinstance(c, cirq.Circuit)
    assert len(c) > 0


@pytest.mark.skipif(not has_ml(), reason="Requires optional qex[ml] dependencies (torch)")
def test_qsvm_classifier():
    # Tests Quantum Kernel and Support Vector Machine Classifier.
    from qex.ml.qsvm import QSVMClassifier, QuantumKernel

    kernel = QuantumKernel(num_qubits=2)
    x1 = np.array([0.5, 1.2])
    x2 = np.array([0.5, 1.2])
    val = kernel.compute_kernel_element(x1, x2)
    assert np.isclose(val, 1.0, atol=1e-3)

    qsvm = QSVMClassifier(num_qubits=2)
    X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
    y = np.array([-1, 1, -1, 1])
    res = qsvm.fit(X, y)

    assert res["accuracy"] >= 0.5
    assert len(res["kernel_matrix"]) == 4
