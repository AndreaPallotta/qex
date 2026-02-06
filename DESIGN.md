# qex Design Specification

## Package Layout

```
qex/
├── __init__.py          # Public API exports
├── experiment.py        # Experiment class (parametric circuit builder)
├── backend.py           # Backend ABC and CirqBackend implementation
├── runner.py            # Runner (executes experiments)
├── store.py             # ResultStore and RunRecord (SQLite persistence)
├── bloch.py             # Bloch sphere visualization (ρ → x,y,z → HTML)
└── demos.py             # Built-in demo experiments
```

## Core Abstractions

### 1. Experiment
- **Purpose**: Define parametric quantum circuits
- **Key**: Name + builder function
- **Constraint**: 1-qubit only for MVP

### 2. Backend
- **Purpose**: Abstract circuit execution interface
- **Concrete**: `CirqBackend` uses Cirq's ideal simulator
- **Output**: Always returns 2x2 density matrix

### 3. Runner
- **Purpose**: Orchestrate experiment execution
- **Flow**: Experiment + params → Circuit → Backend → Density matrix → Artifacts → RunRecord

### 4. ResultStore
- **Purpose**: SQLite persistence for runs
- **Stores**: Metadata, density matrix paths, artifact paths
- **Query**: List runs, filter by experiment name

### 5. Bloch Visualization
- **Math**: `density_matrix_to_bloch(ρ) → (x, y, z)`
- **Artifact**: `bloch_to_html(x, y, z) → HTML string`
- **Output**: Self-contained HTML file viewable in browser

## SQLite Schema

### Tables

**runs**
- `run_id` (TEXT, PRIMARY KEY) - UUID
- `experiment_name` (TEXT, NOT NULL)
- `params` (TEXT, NOT NULL) - JSON
- `backend_name` (TEXT, NOT NULL)
- `timestamp` (REAL, NOT NULL)
- `density_matrix_path` (TEXT, NOT NULL) - Relative path to .npy
- `metadata` (TEXT) - JSON, nullable

**artifacts**
- `artifact_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- `run_id` (TEXT, FOREIGN KEY → runs)
- `artifact_name` (TEXT, NOT NULL) - e.g., "bloch_sphere"
- `artifact_path` (TEXT, NOT NULL) - Relative path
- `artifact_type` (TEXT, NOT NULL) - e.g., "text/html"

### File Organization

```
qex_data/
├── qex.db
├── results/
│   └── {run_id}_rho.npy
└── artifacts/
    └── {run_id}_bloch.html
```

## Bloch Math Contract

**Input**: 2x2 density matrix ρ (complex, Hermitian, Tr=1, positive semidefinite)

**Computation**:
```
x = Tr(ρ · σ_x)
y = Tr(ρ · σ_y)
z = Tr(ρ · σ_z)
```

**Output**: (x, y, z) as real floats, with x² + y² + z² ≤ 1

**HTML Artifact**: Self-contained HTML with 3D Bloch sphere visualization, point marker, and coordinate display.

## Built-in Demo Experiments

1. **X Gate**: |0⟩ → X → |1⟩ (no parameters)
2. **Hadamard**: |0⟩ → H → superposition (no parameters)
3. **Ry Sweep**: |0⟩ → Ry(θ) (parameter: `{"theta": float}`)

## Design Principles

- **Minimal**: Only what's needed for MVP
- **Opinionated**: 1-qubit, ideal simulation, density matrices, SQLite
- **Offline**: No cloud dependencies, no network calls
- **Reproducible**: All parameters and results stored
- **Shippable**: Complete, working MVP

## Constraints (Locked In)

- ✅ Cirq as underlying SDK
- ✅ Fully offline
- ✅ 1-qubit only
- ✅ Ideal simulation (no noise)
- ✅ Density matrices (even if derived from statevectors)
- ✅ SQLite persistence
- ✅ Bloch sphere HTML visualization
- ❌ No multi-qubit support
- ❌ No cloud backends
- ❌ No UI frameworks
- ❌ No vector databases
