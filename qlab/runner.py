"""
Runner: executes experiments with parameters and configuration.
"""

import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import cirq
import numpy as np
from qlab.experiment import Experiment
from qlab.backend import Backend
from qlab.store import RunRecord  # type: ignore
from qlab.bloch import density_matrix_to_bloch, bloch_to_html


class Runner:
    """
    Executes an experiment with given parameters and backend configuration.
    
    The runner coordinates experiment execution, result computation,
    and artifact generation. It does not handle persistence (that's ResultStore's job).
    """
    
    def __init__(self, backend: Backend, base_dir: Optional[Path] = None):
        """
        Initialize a runner with a backend.
        
        Args:
            backend: The backend to use for circuit execution.
            base_dir: Base directory for storing results and artifacts.
                     If None, defaults to current directory.
        """
        self.backend = backend
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        (self.base_dir / "artifacts").mkdir(exist_ok=True)
    
    def run(
        self,
        experiment: Experiment,
        params: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> RunRecord:
        """
        Execute an experiment and return a run record.
        
        Args:
            experiment: The experiment to run.
            params: Parameters for the experiment's circuit builder.
            config: Optional configuration (e.g., qubit selection, metadata).
                   Defaults to using a single qubit.
                   
        Returns:
            RunRecord containing:
            - Experiment name and parameters
            - Density matrix result
            - Generated artifacts (e.g., Bloch sphere HTML)
            - Metadata (timestamp, backend name, etc.)
        """
        config = config or {}
        
        # Generate run ID
        run_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Create qubit (default to GridQubit(0, 0) if not specified)
        qubit = config.get("qubit", cirq.GridQubit(0, 0))
        
        # Build circuit
        circuit = experiment.build_circuit(qubit, params)
        
        # Execute circuit
        rho = self.backend.run(circuit)
        
        # Save density matrix
        rho_path = f"results/{run_id}_rho.npy"
        np.save(self.base_dir / rho_path, rho)
        
        # Generate Bloch sphere visualization
        x, y, z = density_matrix_to_bloch(rho)
        html_content = bloch_to_html(x, y, z, title=f"{experiment.name} - {run_id[:8]}")
        html_path = f"artifacts/{run_id}_bloch.html"
        (self.base_dir / html_path).write_text(html_content)
        
        # Create artifacts dict
        artifacts = {"bloch_sphere": html_path}
        
        # Build metadata
        metadata = config.get("metadata", {})
        metadata["qubit"] = str(qubit)
        
        # Create RunRecord
        record = RunRecord(
            run_id=run_id,
            experiment_name=experiment.name,
            params=params,
            backend_name=self.backend.get_name(),
            timestamp=timestamp,
            density_matrix_path=rho_path,
            artifacts=artifacts,
            metadata=metadata
        )
        
        return record
