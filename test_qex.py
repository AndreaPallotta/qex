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
    
    # Load and check density matrix using the record's path
    rho = np.load(base_dir / record.density_matrix_path)
    print(f"Run ID: {record.run_id}")
    print(f"Density matrix:\n{rho}")
    
    # Check that we get |1⟩ state
    # |1⟩⟨1| = [[0, 0], [0, 1]]
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    if np.allclose(rho, expected):
        print("✓ X gate test passed: Correctly flipped to |1⟩")
    else:
        print("✗ X gate test failed: Density matrix doesn't match |1⟩")
        return False
    
    # Check Bloch coordinates (should be at south pole: z = -1)
    x, y, z = density_matrix_to_bloch(rho)
    print(f"Bloch coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")
    if abs(z + 1.0) < 0.01 and abs(x) < 0.01 and abs(y) < 0.01:
        print("✓ Bloch coordinates correct (south pole)")
    else:
        print("✗ Bloch coordinates incorrect")
        return False
    
    print()
    return True


def test_hadamard():
    """Test Hadamard experiment: |0⟩ → H → superposition"""
    print("Test 2: Hadamard Experiment")
    print("-" * 50)
    
    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = hadamard_experiment()
    
    record = runner.run(experiment, params={})
    
    # Load density matrix using the record's path
    rho = np.load(base_dir / record.density_matrix_path)
    print(f"Run ID: {record.run_id}")
    print(f"Density matrix:\n{rho}")
    
    # Check that we get |+⟩ state
    # |+⟩⟨+| = 0.5 * [[1, 1], [1, 1]]
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    if np.allclose(rho, expected):
        print("✓ Hadamard test passed: Correctly created |+⟩ state")
    else:
        print("✗ Hadamard test failed: Density matrix doesn't match |+⟩")
        return False
    
    # Check Bloch coordinates (should be at +X axis: x = 1)
    x, y, z = density_matrix_to_bloch(rho)
    print(f"Bloch coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")
    if abs(x - 1.0) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01:
        print("✓ Bloch coordinates correct (+X axis)")
    else:
        print("✗ Bloch coordinates incorrect")
        return False
    
    print()
    return True


def test_ry_sweep():
    """Test Ry sweep experiment: |0⟩ → Ry(θ)"""
    print("Test 3: Ry Sweep Experiment")
    print("-" * 50)
    
    base_dir = Path("qex_data")
    backend = CirqBackend()
    runner = Runner(backend, base_dir=base_dir)
    experiment = ry_sweep_experiment()
    
    # Test with theta = π/2 (should create |+⟩ state)
    # Note: Ry(π/2)|0⟩ = |+⟩, not |+i⟩
    theta = np.pi / 2
    record = runner.run(experiment, params={"theta": theta})
    
    # Load density matrix using the record's path
    rho = np.load(base_dir / record.density_matrix_path)
    print(f"Run ID: {record.run_id}")
    print(f"theta = π/2")
    print(f"Density matrix:\n{rho}")
    
    # Ry(π/2)|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
    # |+⟩⟨+| = 0.5 * [[1, 1], [1, 1]]
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    if np.allclose(rho, expected):
        print("✓ Ry sweep test passed: Correctly created |+⟩ state")
    else:
        print("✗ Ry sweep test failed: Density matrix doesn't match |+⟩")
        return False
    
    # Check Bloch coordinates (should be at +X axis: x = 1)
    x, y, z = density_matrix_to_bloch(rho)
    print(f"Bloch coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")
    if abs(x - 1.0) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01:
        print("✓ Bloch coordinates correct (+X axis)")
    else:
        print("✗ Bloch coordinates incorrect")
        return False
    
    print()
    return True


def test_persistence():
    """Test SQLite persistence"""
    print("Test 4: Persistence Test")
    print("-" * 50)
    
    # Clean up any existing database
    db_path = Path("qex_data/qex.db")
    if db_path.exists():
        db_path.unlink()
    
    backend = CirqBackend()
    runner = Runner(backend, base_dir=Path("qex_data"))
    store = ResultStore(db_path)
    
    # Run and save experiments
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
        print(f"Saved: {exp.name} (ID: {record.run_id[:8]}...)")
    
    # Retrieve runs
    print(f"\nTotal runs in database: {len(store.list_runs())}")
    
    # Test filtering
    hadamard_runs = store.list_runs(experiment_name="hadamard")
    print(f"Hadamard runs: {len(hadamard_runs)}")
    if len(hadamard_runs) == 1:
        print("✓ Filtering by experiment name works")
    else:
        print("✗ Filtering failed")
        return False
    
    # Test retrieval by ID
    test_id = saved_ids[0]
    retrieved = store.get_run(test_id)
    if retrieved and retrieved.run_id == test_id:
        print(f"✓ Retrieval by ID works (ID: {test_id[:8]}...)")
    else:
        print("✗ Retrieval by ID failed")
        return False
    
    # Test density matrix loading
    rho = retrieved.get_density_matrix()
    if rho.shape == (2, 2):
        print("✓ Density matrix loading works")
    else:
        print("✗ Density matrix loading failed")
        return False
    
    store.close()
    print()
    return True


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
    print(f"Run ID: {record.run_id}")
    print(f"Density matrix shape: {rho.shape}")

    if rho.shape != (4, 4):
        print(f"✗ Expected 4x4 density matrix, got {rho.shape}")
        return False

    # |Φ⁺⟩⟨Φ⁺| has 1/2 at (0,0), (3,3) and 1/2 at (0,3), (3,0)
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = expected[3, 3] = 0.5
    expected[0, 3] = expected[3, 0] = 0.5
    if np.allclose(rho, expected):
        print("✓ Bell state test passed: density matrix matches |Φ⁺⟩⟨Φ⁺|")
    else:
        print("✗ Bell state test failed: density matrix doesn't match")
        return False

    # Each single-qubit reduced DM should be I/2 (maximally mixed)
    rho_red0 = reduced_density_matrix(rho, 0)
    rho_red1 = reduced_density_matrix(rho, 1)
    half_i = 0.5 * np.eye(2, dtype=complex)
    if np.allclose(rho_red0, half_i) and np.allclose(rho_red1, half_i):
        print("✓ Reduced density matrices are I/2 (maximally mixed)")
    else:
        print("✗ Reduced density matrices incorrect")
        return False

    print()
    return True


def test_advanced_features():
    """Test purity, entropy, NoisyCirqBackend, and state fidelity calculation"""
    print("Test 6: Advanced Metrics, Noisy Simulation & Fidelity")
    print("-" * 50)
    
    # 1. Test metrics on pure state
    rho_pure = np.array([[0, 0], [0, 1]], dtype=complex) # |1><1|
    purity_pure = state_purity(rho_pure)
    entropy_pure = state_entropy(rho_pure)
    print(f"Pure state (|1⟩) -> Purity: {purity_pure:.4f}, Entropy: {entropy_pure:.4f}")
    if not np.isclose(purity_pure, 1.0) or not np.isclose(entropy_pure, 0.0):
        print("✗ Pure state metrics incorrect")
        return False
    print("✓ Pure state metrics correct")

    # 2. Test metrics on maximally mixed state
    rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex) # I/2
    purity_mixed = state_purity(rho_mixed)
    entropy_mixed = state_entropy(rho_mixed)
    print(f"Mixed state (I/2) -> Purity: {purity_mixed:.4f}, Entropy: {entropy_mixed:.4f}")
    if not np.isclose(purity_mixed, 0.5) or not np.isclose(entropy_mixed, 1.0):
        print("✗ Mixed state metrics incorrect")
        return False
    print("✓ Mixed state metrics correct")

    # 3. Test Noisy Backend simulation
    exp = hadamard_experiment()
    backend_ideal = CirqBackend()
    backend_noisy = NoisyCirqBackend(p=0.1) # 10% depolarizing noise
    
    rho_ideal = backend_ideal.run(exp.build_circuit([cirq.GridQubit(0, 0)], {}))
    rho_noisy = backend_noisy.run(exp.build_circuit([cirq.GridQubit(0, 0)], {}))
    
    purity_noisy = state_purity(rho_noisy)
    entropy_noisy = state_entropy(rho_noisy)
    
    print(f"Noisy Hadamard state -> Purity: {purity_noisy:.4f}, Entropy: {entropy_noisy:.4f}")
    if purity_noisy >= 1.0 or entropy_noisy <= 0.0:
        print("✗ Noisy state metrics failed to show mixed properties")
        return False
    print("✓ Noisy backend correctly produced mixed state")

    # 4. Test State Fidelity
    fid_self = state_fidelity(rho_ideal, rho_ideal)
    fid_orthogonal = state_fidelity(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))
    fid_noisy = state_fidelity(rho_ideal, rho_noisy)
    
    print(f"Fidelity(ideal, ideal): {fid_self:.4f}")
    print(f"Fidelity(|0⟩, |1⟩): {fid_orthogonal:.4f}")
    print(f"Fidelity(ideal, noisy): {fid_noisy:.4f}")
    
    if not np.isclose(fid_self, 1.0):
        print("✗ Fidelity of self-state must be 1.0")
        return False
    if not np.isclose(fid_orthogonal, 0.0):
        print("✗ Fidelity of orthogonal states must be 0.0")
        return False
    if fid_noisy <= 0.0 or fid_noisy >= 1.0:
        print("✗ Noisy fidelity must be strictly in (0, 1)")
        return False
    print("✓ State fidelity calculations correct")
    
    # 5. Test 10-qubit GHZ state simulation
    print("Simulating 10-qubit GHZ state...")
    qubits_10 = [cirq.GridQubit(0, i) for i in range(10)]
    exp_ghz = bell_state_experiment()
    rho_ghz = backend_ideal.run(exp_ghz.build_circuit(qubits_10, {}))
    print(f"10-qubit GHZ density matrix shape: {rho_ghz.shape}")
    if rho_ghz.shape != (1024, 1024):
        print("✗ 10-qubit GHZ simulation returned wrong density matrix shape")
        return False
    print("✓ 10-qubit GHZ simulation successful")
    
    print()
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("qex Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_x_gate,
        test_hadamard,
        test_ry_sweep,
        test_persistence,
        test_bell_state,
        test_advanced_features,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
