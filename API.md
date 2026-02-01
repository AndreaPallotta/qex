# qlab Public API

## Package Structure

```
qlab/
├── __init__.py          # Public exports
├── experiment.py        # Experiment class
├── backend.py           # Backend interface and CirqBackend
├── runner.py            # Runner class
├── store.py             # ResultStore and RunRecord
├── bloch.py             # Bloch sphere visualization
└── demos.py             # Built-in demo experiments
```

## Core Abstractions

### Experiment

Defines a parametric circuit builder.

```python
class Experiment:
    def __init__(name: str, builder: Callable[[cirq.Qid, Dict[str, Any]], cirq.Circuit])
    def build_circuit(qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit
```

### Backend

Abstract interface for circuit execution.

```python
class Backend(ABC):
    @abstractmethod
    def run(circuit: cirq.Circuit) -> np.ndarray  # Returns 2x2 density matrix
    @abstractmethod
    def get_name() -> str

class CirqBackend(Backend):
    def __init__()
    def run(circuit: cirq.Circuit) -> np.ndarray
    def get_name() -> str  # Returns "cirq_ideal"
```

### Runner

Executes experiments and generates results/artifacts.

```python
class Runner:
    def __init__(backend: Backend)
    def run(experiment: Experiment, params: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> RunRecord
```

### ResultStore

SQLite persistence layer.

```python
class RunRecord:
    def __init__(run_id: str, experiment_name: str, params: Dict[str, Any], 
                 backend_name: str, timestamp: float, density_matrix_path: str,
                 artifacts: Dict[str, str], metadata: Optional[Dict[str, Any]] = None)
    def get_density_matrix() -> np.ndarray

class ResultStore:
    def __init__(db_path: Path)
    def save_run(record: RunRecord) -> None
    def get_run(run_id: str) -> Optional[RunRecord]
    def list_runs(experiment_name: Optional[str] = None, limit: Optional[int] = None) -> List[RunRecord]
    def close() -> None
```

### Bloch Visualization

Density matrix to Bloch sphere coordinates and HTML.

```python
def density_matrix_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]
def bloch_to_html(x: float, y: float, z: float, title: str = "Bloch Sphere") -> str
```

### Demo Experiments

Built-in validation experiments.

```python
def x_gate_experiment() -> Experiment
def hadamard_experiment() -> Experiment
def ry_sweep_experiment() -> Experiment  # Takes {"theta": float} parameter
```

## Usage Pattern

```python
from qlab import Experiment, CirqBackend, Runner, ResultStore
from qlab.demos import x_gate_experiment

# Setup
backend = CirqBackend()
runner = Runner(backend)
store = ResultStore(Path("qlab_data/qlab.db"))

# Run experiment
experiment = x_gate_experiment()
record = runner.run(experiment, params={})

# Persist
store.save_run(record)

# Retrieve
runs = store.list_runs(experiment_name="x_gate")
store.close()
```
