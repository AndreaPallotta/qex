"""
Runner: executes experiments with parameters and configuration.
"""

import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import cirq
import numpy as np
from qex.experiment import Experiment
from qex.backend import Backend
from qex.store import RunRecord  # type: ignore
from qex.bloch import density_matrix_to_bloch, bloch_to_html, reduced_density_matrix


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
        config: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        """
        Execute an experiment and return a run record.

        Args:
            experiment: The experiment to run.
            params: Parameters for the experiment's circuit builder.
            config: Optional configuration. Use "qubits" (list of cirq.Qid) to set
                   qubits; defaults to [GridQubit(0, 0)].

        Returns:
            RunRecord containing density matrix path, artifacts, and metadata.
        """
        config = config or {}

        run_id = str(uuid.uuid4())
        timestamp = time.time()

        qubits = config.get("qubits")
        if qubits is None:
            qubits = [cirq.GridQubit(0, 0)]
        elif isinstance(qubits, cirq.Qid):
            qubits = [qubits]
        else:
            qubits = list(qubits)

        circuit = experiment.build_circuit(qubits, params)
        rho = self.backend.run(circuit)

        rho_path = f"results/{run_id}_rho.npy"
        np.save(self.base_dir / rho_path, rho)

        artifacts: Dict[str, str] = {}
        if rho.shape == (2, 2):
            x, y, z = density_matrix_to_bloch(rho)
            html_content = bloch_to_html(
                x, y, z, title=f"{experiment.name} - {run_id[:8]}"
            )
            html_path = f"artifacts/{run_id}_bloch.html"
            (self.base_dir / html_path).write_text(html_content)
            artifacts["bloch_sphere"] = html_path
        else:
            rho_red = reduced_density_matrix(rho, qubit_index=0)
            x, y, z = density_matrix_to_bloch(rho_red)
            html_content = bloch_to_html(
                x, y, z,
                title=f"{experiment.name} (qubit 0) - {run_id[:8]}",
            )
            html_path = f"artifacts/{run_id}_bloch_qubit0.html"
            (self.base_dir / html_path).write_text(html_content)
            artifacts["bloch_sphere_qubit0"] = html_path

        metadata = dict(config.get("metadata", {}))
        metadata["qubits"] = [str(q) for q in qubits]

        record = RunRecord(
            run_id=run_id,
            experiment_name=experiment.name,
            params=params,
            backend_name=self.backend.get_name(),
            timestamp=timestamp,
            density_matrix_path=rho_path,
            artifacts=artifacts,
            metadata=metadata,
        )
        return record
