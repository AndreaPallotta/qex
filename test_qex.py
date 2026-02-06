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

from qex import CirqBackend, Runner, ResultStore
from qex.demos import (
    x_gate_experiment,
    hadamard_experiment,
    ry_sweep_experiment,
    bell_state_experiment,
)
from qex.bloch import density_matrix_to_bloch, reduced_density_matrix


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
