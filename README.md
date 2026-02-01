# qlab

<div align="center">
  <img src="https://raw.githubusercontent.com/AndreaPallotta/qlab/main/assets/logo.png" alt="qlab logo" width="200"/>
</div>

A lightweight experiment-runner and lab notebook for quantum computing, built on top of Cirq.

## Overview

`qlab` is designed for running quantum experiments, tracking results, and visualizing outcomes. It focuses on reproducibility, persistence, and visualization rather than being a general-purpose quantum framework.

**Key Features:**
- 🧪 **Experiment Management**: Define parametric quantum circuits and run them systematically
- 💾 **Result Persistence**: Store runs, results, and metadata in SQLite
- 📊 **Visualization**: Automatic Bloch sphere visualization for 1-qubit states
- 🔬 **Reproducibility**: All parameters and results are stored for later analysis
- 🚀 **Simple API**: Minimal, opinionated design focused on experiments

## Installation

```bash
pip install qex
```

## Quick Start

```python
from qlab import CirqBackend, Runner, ResultStore
from qlab.demos import hadamard_experiment
from pathlib import Path

# Setup
backend = CirqBackend()
runner = Runner(backend)
store = ResultStore(Path("qlab_data/qlab.db"))

# Run an experiment
experiment = hadamard_experiment()
record = runner.run(experiment, params={})

# Persist results
store.save_run(record)

# Retrieve and view
runs = store.list_runs(experiment_name="hadamard")
rho = runs[0].get_density_matrix()
print(f"Density matrix:\n{rho}")

store.close()
```

## Built-in Experiments

`qlab` includes three demo experiments for validation:

- **X Gate**: `|0⟩ → X → |1⟩` - Simple bit flip
- **Hadamard**: `|0⟩ → H → superposition` - Creates equal superposition
- **Ry Sweep**: `|0⟩ → Ry(θ)` - Rotation around Y-axis with parameter `theta`

## Documentation

- **[API Reference](API.md)**: Complete API documentation
- **[Design Specification](DESIGN.md)**: Architecture and design decisions
- **[Database Schema](SCHEMA.md)**: SQLite schema documentation
- **[Bloch Sphere Contract](BLOCH_CONTRACT.md)**: Visualization math and interface

## Current Scope (MVP)

- ✅ 1-qubit experiments only
- ✅ Ideal simulation (no noise)
- ✅ Density matrix results
- ✅ SQLite persistence
- ✅ Bloch sphere HTML visualization
- ✅ Cirq-based backend

## Requirements

- Python >= 3.12
- Cirq >= 1.6.1
- NumPy >= 1.24.0

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please see the [Design Specification](DESIGN.md) for architecture details and constraints.
