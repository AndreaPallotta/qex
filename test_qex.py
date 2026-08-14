#!/usr/bin/env python3
"""
Test script for qex - runs basic validation tests.
"""

import sys
from pathlib import Path
import cirq
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qex import CirqBackend, Runner, ResultStore, NoisyCirqBackend
from qex.demos import (
    x_gate_experiment,
    hadamard_experiment,
    ry_sweep_experiment,
    bell_state_experiment,
    quantum_teleportation_experiment,
    ghz_state_experiment,
)
from qex.bloch import (
    density_matrix_to_bloch,
    reduced_density_matrix,
    state_purity,
    state_entropy,
    state_fidelity,
)


def test_x_gate():
    """Test X gate experiment: |0⟩ → X → |1⟩"""
    print("Test 1: X Gate Experiment")
    print("-" * 50)

    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = x_gate_experiment()

    record = runner.run(experiment, params={})

    rho = np.load(base_dir / record.density_matrix_path)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(rho, expected)

    x, y, z = density_matrix_to_bloch(rho)
    assert abs(z + 1.0) < 0.01 and abs(x) < 0.01 and abs(y) < 0.01


def test_hadamard():
    """Test Hadamard experiment: |0⟩ → H → superposition"""
    print("Test 2: Hadamard Experiment")
    print("-" * 50)

    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = hadamard_experiment()

    record = runner.run(experiment, params={})

    rho = np.load(base_dir / record.density_matrix_path)
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    assert np.allclose(rho, expected)

    x, y, z = density_matrix_to_bloch(rho)
    assert abs(x - 1.0) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01


def test_ry_sweep():
    """Test Ry sweep experiment: |0⟩ → Ry(θ)"""
    print("Test 3: Ry Sweep Experiment")
    print("-" * 50)

    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = ry_sweep_experiment()

    theta = np.pi / 2
    record = runner.run(experiment, params={"theta": theta})

    rho = np.load(base_dir / record.density_matrix_path)
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    assert np.allclose(rho, expected)

    x, y, z = density_matrix_to_bloch(rho)
    assert abs(x - 1.0) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01


def test_persistence():
    """Test SQLite persistence"""
    print("Test 4: Persistence Test")
    print("-" * 50)

    db_path = Path("qex_data/qex.db")
    if db_path.exists():
        db_path.unlink()

    backend = CirqBackend()
    runner = Runner(backend, base_dir=Path("qex_data"))
    store = ResultStore(db_path)

    experiments = [
        (x_gate_experiment(), {}),
        (hadamard_experiment(), {}),
        (ry_sweep_experiment(), {"theta": np.pi / 4}),
    ]

    saved_ids = []
    for exp, params in experiments:
        record = runner.run(exp, params)
        store.save_run(record)
        saved_ids.append(record.run_id)

    assert len(store.list_runs()) == 3

    hadamard_runs = store.list_runs(experiment_name="hadamard")
    assert len(hadamard_runs) == 1

    test_id = saved_ids[0]
    retrieved = store.get_run(test_id)
    assert retrieved is not None and retrieved.run_id == test_id

    rho = retrieved.get_density_matrix()
    assert rho.shape == (2, 2)

    store.close()


def test_bell_state():
    """Test Bell state experiment: 2-qubit |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    print("Test 5: Bell State (multi-qubit)")
    print("-" * 50)

    base_dir = Path("qex_data")
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = bell_state_experiment()

    record = runner.run(
        experiment,
        params={},
        config={"qubits": qubits},
    )

    rho = np.load(base_dir / record.density_matrix_path)
    assert rho.shape == (4, 4)

    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = expected[3, 3] = 0.5
    expected[0, 3] = expected[3, 0] = 0.5
    assert np.allclose(rho, expected)

    rho_red0 = reduced_density_matrix(rho, 0)
    rho_red1 = reduced_density_matrix(rho, 1)
    half_i = 0.5 * np.eye(2, dtype=complex)
    assert np.allclose(rho_red0, half_i) and np.allclose(rho_red1, half_i)


def test_advanced_features():
    """Test purity, entropy, NoisyCirqBackend, and state fidelity calculation"""
    print("Test 6: Advanced Metrics, Noisy Simulation & Fidelity")
    print("-" * 50)

    rho_pure = np.array([[0, 0], [0, 1]], dtype=complex)
    purity_pure = state_purity(rho_pure)
    entropy_pure = state_entropy(rho_pure)
    assert np.isclose(purity_pure, 1.0) and np.isclose(entropy_pure, 0.0)

    rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    purity_mixed = state_purity(rho_mixed)
    entropy_mixed = state_entropy(rho_mixed)
    assert np.isclose(purity_mixed, 0.5) and np.isclose(entropy_mixed, 1.0)

    exp = hadamard_experiment()
    backend_ideal = CirqBackend()
    backend_noisy = NoisyCirqBackend(p=0.1)

    rho_ideal = backend_ideal.run(exp.build_circuit([cirq.GridQubit(0, 0)], {}))
    rho_noisy = backend_noisy.run(exp.build_circuit([cirq.GridQubit(0, 0)], {}))

    purity_noisy = state_purity(rho_noisy)
    entropy_noisy = state_entropy(rho_noisy)
    assert purity_noisy < 1.0 and entropy_noisy > 0.0

    fid_self = state_fidelity(rho_ideal, rho_ideal)
    fid_orthogonal = state_fidelity(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))
    fid_noisy = state_fidelity(rho_ideal, rho_noisy)

    assert np.isclose(fid_self, 1.0)
    assert np.isclose(fid_orthogonal, 0.0)
    assert 0.0 < fid_noisy < 1.0

    qubits_10 = [cirq.GridQubit(0, i) for i in range(10)]
    exp_ghz = bell_state_experiment()
    rho_ghz = backend_ideal.run(exp_ghz.build_circuit(qubits_10, {}))
    assert rho_ghz.shape == (1024, 1024)


def test_teleportation():
    # Test 3-qubit quantum teleportation experiment
    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    exp = quantum_teleportation_experiment()
    q = [cirq.GridQubit(0, i) for i in range(3)]
    record = runner.run(exp, params={"theta": 1.2}, config={"qubits": q})
    rho = np.load(base_dir / record.density_matrix_path)
    assert rho.shape == (8, 8)
    assert np.isclose(np.trace(rho), 1.0)


def test_ghz_state():
    # Test N-qubit GHZ state creation
    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    exp = ghz_state_experiment()
    q = [cirq.GridQubit(0, i) for i in range(3)]
    record = runner.run(exp, params={}, config={"qubits": q})
    rho = np.load(base_dir / record.density_matrix_path)
    assert rho.shape == (8, 8)
    assert np.isclose(rho[0, 0], 0.5)
    assert np.isclose(rho[7, 7], 0.5)


def main():
    """Run all tests standalone."""
    tests = [
        test_x_gate,
        test_hadamard,
        test_ry_sweep,
        test_persistence,
        test_bell_state,
        test_advanced_features,
        test_teleportation,
        test_ghz_state,
    ]
    for test in tests:
        test()
    print("All standalone tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
