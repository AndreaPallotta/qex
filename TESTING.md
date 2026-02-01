# Testing qlab

## Quick Start

Run the test suite:

```bash
./run_tests.sh
```

Or run the test script directly:

```bash
python test_qlab.py
```

## Test Cases

The test suite includes four validation tests:

### Test 1: X Gate Experiment
- Runs: `|0⟩ → X → |1⟩`
- Validates: Density matrix matches `|1⟩⟨1|`
- Validates: Bloch coordinates at south pole (z = -1)

### Test 2: Hadamard Experiment
- Runs: `|0⟩ → H → |+⟩`
- Validates: Density matrix matches `|+⟩⟨+|`
- Validates: Bloch coordinates at +X axis (x = 1)

### Test 3: Ry Sweep Experiment
- Runs: `|0⟩ → Ry(π/2) → |+i⟩`
- Validates: Density matrix matches `|+i⟩⟨+i|`
- Validates: Bloch coordinates at +Y axis (y = 1)

### Test 4: Persistence Test
- Tests SQLite database operations
- Validates: Saving runs
- Validates: Retrieving runs by ID
- Validates: Filtering by experiment name
- Validates: Loading density matrices from disk

## Manual Testing

You can also test qlab interactively:

```python
from qlab import CirqBackend, Runner, ResultStore
from qlab.demos import hadamard_experiment
from pathlib import Path

# Setup
backend = CirqBackend()
runner = Runner(backend, base_dir=Path("qlab_data"))
store = ResultStore(Path("qlab_data/qlab.db"))

# Run experiment
experiment = hadamard_experiment()
record = runner.run(experiment, params={})

# Save
store.save_run(record)

# View results
print(f"Run ID: {record.run_id}")
print(f"Bloch HTML: qlab_data/{record.artifacts['bloch_sphere']}")

# Retrieve
runs = store.list_runs(experiment_name="hadamard")
rho = runs[0].get_density_matrix()
print(f"Density matrix:\n{rho}")

store.close()
```

## Expected Output

After running tests, you should see:
- `qlab_data/` directory with:
  - `qlab.db` - SQLite database
  - `results/` - Density matrix .npy files
  - `artifacts/` - Bloch sphere HTML files

Open any HTML file in `artifacts/` to view the 3D Bloch sphere visualization.
