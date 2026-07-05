# qex

<div align="center">
  <img src="https://raw.githubusercontent.com/AndreaPallotta/qex/main/assets/logo.jpg" alt="qex logo" width="200"/>
  <p><strong>A lightweight experiment-runner and lab notebook for quantum computing, built on top of Cirq.</strong></p>
</div>

---

`qex` decouples your quantum experiment definitions from execution backends, persisting your runs in a local SQLite database and providing a visual web dashboard to analyze state metrics, density matrices, and compare state fidelities side-by-side.

## Key Features

- 🧪 **Experiment Abstraction**: Define parameterized quantum circuits independent of physical backends.
- 💾 **SQLite Lab Notebook**: All runs, timestamps, parameters, backends, and density matrices are tracked and stored automatically.
- 📉 **Noisy Simulation Backend**: Run ideal simulations or inject custom physical depolarizing noise.
- 📐 **Quantum State Metrics**: Real-time evaluation of **Purity** and **Von Neumann Entropy** for simulated mixed states.
- 📊 **Interactive Web Dashboard**: Beautiful dark-mode SPA visualizer containing a 3D Bloch sphere (Three.js), density matrix grids, and run creators.
- ⚖️ **Side-by-Side Run Comparison**: Compare any two runs side-by-side and compute their quantum **State Fidelity**.
- 💻 **Robust CLI Tool**: Start the UI server, list runs, execute experiments, and open visualizations from the terminal.
- 📡 **Multi-Qubit Scaling**: Simulate circuits from 1 up to **10 qubits** ($1024 \times 1024$ density matrix dimension) with grid rendering performance optimization.

## Installation

```bash
pip install qex
```

## Quick Start

### 1. Define and Run an Experiment in Python
```python
import cirq
from qex import CirqBackend, Runner, ResultStore, Experiment

# 1. Define a parameterized experiment (e.g. Hadamard on N qubits)
def multi_hadamard():
    def builder(qubits, params):
        return cirq.Circuit(cirq.H(q) for q in qubits)
    return Experiment(name="multi_hadamard", builder=builder)

# 2. Setup the runner and database store
backend = CirqBackend()
runner = Runner(backend, base_dir="qex_data")
store = ResultStore("qex_data/qex.db")

# 3. Simulate a 3-qubit execution
qubits = [cirq.GridQubit(0, i) for i in range(3)]
record = runner.run(multi_hadamard(), params={}, config={"qubits": qubits})

# 4. Save to your lab notebook
store.save_run(record)
store.close()
```

### 2. Control via CLI
```bash
# List all executed runs in the notebook
qex list

# Execute a new run with depolarizing noise (p = 0.05) on 4 qubits
qex run hadamard --qubits 4 --noise 0.05

# Open a 3D Bloch Sphere visualization in the browser
qex view <run_id>
```

### 3. Open the Interactive Web Dashboard
```bash
qex ui
```
This launches a local server on port 8000. Use the visual dashboard to run new circuits, view 3D Bloch vectors, analyze purity/entropy, and toggle **Compare Mode** to check quantum state fidelity between different ideal and noisy runs.

## Requirements

- Python >= 3.12
- Cirq >= 1.6.1
- NumPy >= 1.24.0

## License

Apache License 2.0

## FYI / Disclaimer

> [!NOTE]
> The author is not a professional physicist. This library is built as an exploratory workspace, educational tool, and local lab notebook. It may contain bugs, experimental behaviors, or mathematical/physical inconsistencies. Contributions and corrections are highly welcome!
