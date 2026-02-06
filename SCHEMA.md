# SQLite Schema for qex

## Overview

The database stores run metadata, references to density matrix files, and artifact paths.
Density matrices themselves are stored as `.npy` files in a `results/` directory.
Artifacts (HTML visualizations) are stored in an `artifacts/` directory.

## Tables

### `runs`

Stores metadata for each experiment run.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `run_id` | TEXT | PRIMARY KEY | Unique identifier (UUID v4) |
| `experiment_name` | TEXT | NOT NULL | Name of the experiment |
| `params` | TEXT | NOT NULL | JSON-encoded parameters dictionary |
| `backend_name` | TEXT | NOT NULL | Name of the backend used |
| `timestamp` | REAL | NOT NULL | Unix timestamp (seconds since epoch) |
| `density_matrix_path` | TEXT | NOT NULL | Relative path to .npy file (from db directory) |
| `metadata` | TEXT | | JSON-encoded additional metadata (nullable) |

### `artifacts`

Stores references to generated artifacts (e.g., Bloch sphere HTML).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `artifact_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Auto-incrementing ID |
| `run_id` | TEXT | NOT NULL, FOREIGN KEY → runs(run_id) | Reference to the run |
| `artifact_name` | TEXT | NOT NULL | Name of artifact (e.g., "bloch_sphere") |
| `artifact_path` | TEXT | NOT NULL | Relative path to artifact file |
| `artifact_type` | TEXT | NOT NULL | MIME type or file type (e.g., "text/html") |

## Indexes

- Index on `runs.experiment_name` for fast filtering
- Index on `runs.timestamp` for chronological ordering
- Index on `artifacts.run_id` for fast artifact lookup

## File Organization

```
qex_data/
├── qex.db               # SQLite database
├── results/
│   ├── {run_id}_rho.npy # Density matrices
│   └── ...
└── artifacts/
    ├── {run_id}_bloch.html  # Bloch sphere visualizations
    └── ...
```

## Notes

- All file paths stored in the database are relative to the database directory.
- Density matrices are stored as numpy arrays with complex dtype.
- Artifacts are stored as files (HTML, etc.) and referenced by path.
- The database directory structure is created automatically on first use.
