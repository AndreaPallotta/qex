"""
ResultStore: SQLite persistence for runs, results, and artifacts.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import sqlite3
import numpy as np
import uuid


class RunRecord:
    """
    Metadata and references for a single experiment run.
    
    Contains:
    - Run ID (UUID or auto-increment)
    - Experiment name
    - Parameters dictionary (JSON-serializable)
    - Backend name
    - Timestamp
    - Path to density matrix file (numpy .npy)
    - Path to artifacts (e.g., Bloch sphere HTML)
    - Additional metadata
    """
    
    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        params: Dict[str, Any],
        backend_name: str,
        timestamp: float,
        density_matrix_path: str,
        artifacts: Dict[str, str],  # artifact_name -> file_path
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a run record.
        
        Args:
            run_id: Unique identifier for this run.
            experiment_name: Name of the experiment.
            params: Parameters used for this run.
            backend_name: Name of the backend used.
            timestamp: Unix timestamp of when run was executed.
            density_matrix_path: Path to saved density matrix (.npy file).
            artifacts: Dictionary mapping artifact names to file paths.
            metadata: Optional additional metadata.
        """
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.params = params
        self.backend_name = backend_name
        self.timestamp = timestamp
        self.density_matrix_path = density_matrix_path
        self.artifacts = artifacts
        self.metadata = metadata or {}
        self._base_dir: Optional[Path] = None  # Set by ResultStore when loading
    
    def set_base_dir(self, base_dir: Path) -> None:
        """Set the base directory for resolving relative paths."""
        self._base_dir = base_dir
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Load and return the density matrix for this run.
        
        Returns:
            2x2 density matrix (complex dtype).
        """
        if self._base_dir is None:
            raise ValueError("Base directory not set. Use ResultStore.get_run() to load records.")
        full_path = self._base_dir / self.density_matrix_path
        return np.load(full_path)


class ResultStore:
    """
    SQLite-based persistence layer for experiment runs.
    
    Stores run metadata, density matrices, and artifact references.
    Provides query interface for retrieving runs.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the result store.
        
        Args:
            db_path: Path to SQLite database file.
                    If file doesn't exist, it will be created with schema.
        """
        self.db_path = Path(db_path).resolve()
        self.base_dir = self.db_path.parent
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()
        self._ensure_directories()
    
    def _initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                params TEXT NOT NULL,
                backend_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                density_matrix_path TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create artifacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_experiment_name 
            ON runs(experiment_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
            ON runs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifacts_run_id 
            ON artifacts(run_id)
        """)
        
        self.conn.commit()
    
    def _ensure_directories(self) -> None:
        """Create results and artifacts directories if they don't exist."""
        (self.base_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    
    def save_run(self, record: RunRecord) -> None:
        """
        Persist a run record to the database.
        
        Args:
            record: The RunRecord to save.
        """
        cursor = self.conn.cursor()
        
        # Save run metadata
        cursor.execute("""
            INSERT INTO runs (run_id, experiment_name, params, backend_name, 
                           timestamp, density_matrix_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_id,
            record.experiment_name,
            json.dumps(record.params),
            record.backend_name,
            record.timestamp,
            record.density_matrix_path,
            json.dumps(record.metadata) if record.metadata else None
        ))
        
        # Save artifacts
        for artifact_name, artifact_path in record.artifacts.items():
            # Determine artifact type from extension
            artifact_type = "text/html" if artifact_path.endswith(".html") else "application/octet-stream"
            cursor.execute("""
                INSERT INTO artifacts (run_id, artifact_name, artifact_path, artifact_type)
                VALUES (?, ?, ?, ?)
            """, (record.run_id, artifact_name, artifact_path, artifact_type))
        
        self.conn.commit()
    
    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a run record by ID.
        
        Args:
            run_id: The run ID to look up.
            
        Returns:
            RunRecord if found, None otherwise.
        """
        cursor = self.conn.cursor()
        
        # Get run metadata
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        
        # Get artifacts
        cursor.execute("""
            SELECT artifact_name, artifact_path 
            FROM artifacts 
            WHERE run_id = ?
        """, (run_id,))
        artifacts = {row["artifact_name"]: row["artifact_path"] for row in cursor.fetchall()}
        
        # Build RunRecord
        record = RunRecord(
            run_id=row["run_id"],
            experiment_name=row["experiment_name"],
            params=json.loads(row["params"]),
            backend_name=row["backend_name"],
            timestamp=row["timestamp"],
            density_matrix_path=row["density_matrix_path"],
            artifacts=artifacts,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )
        record.set_base_dir(self.base_dir)
        return record
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[RunRecord]:
        """
        List run records, optionally filtered by experiment name.
        
        Args:
            experiment_name: Optional filter by experiment name.
            limit: Optional maximum number of records to return.
            
        Returns:
            List of RunRecord objects, ordered by timestamp (newest first).
        """
        cursor = self.conn.cursor()
        
        query = "SELECT run_id FROM runs"
        params = []
        if experiment_name:
            query += " WHERE experiment_name = ?"
            params.append(experiment_name)
        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        run_ids = [row["run_id"] for row in cursor.fetchall()]
        
        records = []
        for run_id in run_ids:
            record = self.get_run(run_id)
            if record:
                records.append(record)
        
        return records
    
    def close(self) -> None:
        """
        Close the database connection.
        """
        self.conn.close()
