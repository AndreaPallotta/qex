"""
Web UI dashboard server for qex.
Serves dashboard.html and a JSON API using python's built-in http.server.
"""

import http.server
import json
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from qex.backend import CirqBackend
from qex.runner import Runner
from qex.store import ResultStore
from qex.demos import (
    x_gate_experiment,
    hadamard_experiment,
    ry_sweep_experiment,
    bell_state_experiment,
)
from qex.bloch import density_matrix_to_bloch, reduced_density_matrix

# Global DB Path reference for the server handler
_db_path: Path = Path("qex_data/qex.db")


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the qex dashboard."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard logging to keep console clean unless requested
        pass

    def do_GET(self) -> None:
        """Handle GET requests for files and JSON API."""
        parsed_path = self.path.split("?")[0]

        # Route: Serve HTML Dashboard
        if parsed_path in ("/", "/index.html", "/dashboard"):
            self.serve_dashboard()
            return

        # Route: API - Get all runs
        if parsed_path == "/api/runs":
            self.api_get_runs()
            return

        # Route: API - Get density matrix for a run
        # Path format: /api/runs/<run_id>/density_matrix
        if parsed_path.startswith("/api/runs/") and parsed_path.endswith("/density_matrix"):
            parts = parsed_path.split("/")
            if len(parts) >= 5:
                run_id = parts[3]
                self.api_get_density_matrix(run_id)
                return

        # Route: API - Compare two runs
        if parsed_path == "/api/compare":
            self.api_compare_runs()
            return

        # Fallback 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self) -> None:
        """Handle POST requests for API actions."""
        parsed_path = self.path.split("?")[0]

        # Route: API - Run experiment
        if parsed_path == "/api/run":
            self.api_run_experiment()
            return

        # Fallback 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def serve_dashboard(self) -> None:
        """Serve the dashboard.html static file."""
        html_file = Path(__file__).parent / "dashboard.html"
        if not html_file.exists():
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Error: dashboard.html not found.")
            return

        content = html_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def api_get_runs(self) -> None:
        """Retrieve and return all runs from SQLite database."""
        try:
            store = ResultStore(_db_path)
            runs = store.list_runs()
            store.close()
        except Exception as e:
            self.send_json_error(f"Database error: {str(e)}", 500)
            return

        result: List[Dict[str, Any]] = []
        for r in runs:
            result.append({
                "run_id": r.run_id,
                "experiment_name": r.experiment_name,
                "params": r.params,
                "backend_name": r.backend_name,
                "timestamp": r.timestamp,
                "density_matrix_path": r.density_matrix_path,
                "artifacts": r.artifacts,
                "metadata": r.metadata,
            })

        self.send_json(result)

    def api_get_density_matrix(self, run_id: str) -> None:
        """Get the density matrix values and Bloch sphere coords for a run."""
        try:
            store = ResultStore(_db_path)
            record = store.get_run(run_id)
            store.close()
        except Exception as e:
            self.send_json_error(f"Database error: {str(e)}", 500)
            return

        if not record:
            self.send_json_error("Run not found", 404)
            return

        try:
            rho = record.get_density_matrix()
        except Exception as e:
            self.send_json_error(f"Error loading density matrix: {str(e)}", 500)
            return

        # Format density matrix as a list of lists of complex components
        matrix_list: List[List[Dict[str, float]]] = []
        for row in rho:
            row_list: List[Dict[str, float]] = []
            for val in row:
                row_list.append({
                    "real": float(np.real(val)),
                    "imag": float(np.imag(val))
                })
            matrix_list.append(row_list)

        # Compute Bloch coordinates if applicable (1-qubit or qubit 0 reduced state)
        bloch_data = None
        purity = 1.0
        entropy = 0.0
        try:
            from qex.bloch import state_purity, state_entropy
            purity = state_purity(rho)
            entropy = state_entropy(rho)
            
            if rho.shape == (2, 2):
                x, y, z = density_matrix_to_bloch(rho)
                bloch_data = {"x": x, "y": y, "z": z}
            elif rho.shape == (4, 4):
                # 2-qubit system: compute reduced density matrix of qubit 0
                rho_red = reduced_density_matrix(rho, 0)
                x, y, z = density_matrix_to_bloch(rho_red)
                bloch_data = {"x": x, "y": y, "z": z}
        except Exception:
            # Non-critical: ignore if Bloch coords calculations fail (e.g. 3+ qubits)
            pass

        self.send_json({
            "matrix": matrix_list,
            "bloch": bloch_data,
            "purity": purity,
            "entropy": entropy
        })

    def api_run_experiment(self) -> None:
        """Run an experiment with parameters and persist it."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self.send_json_error(f"Invalid request payload: {str(e)}", 400)
            return

        exp_name = payload.get("experiment")
        params = payload.get("params", {})
        noise = float(payload.get("noise", 0.0))
        num_qubits = int(payload.get("qubits", 2 if exp_name == "bell_state" else 1))

        if not 1 <= num_qubits <= 10:
            self.send_json_error("Qubits count must be between 1 and 10", 400)
            return

        import cirq
        config = {"qubits": [cirq.GridQubit(0, i) for i in range(num_qubits)]}

        # Resolve experiment
        if exp_name == "x_gate":
            exp = x_gate_experiment()
        elif exp_name == "hadamard":
            exp = hadamard_experiment()
        elif exp_name == "ry_sweep":
            exp = ry_sweep_experiment()
            if "theta" not in params:
                params["theta"] = np.pi / 2
        elif exp_name == "bell_state":
            if num_qubits < 2:
                self.send_json_error("Bell/GHZ state experiment requires at least 2 qubits", 400)
                return
            exp = bell_state_experiment()
        else:
            self.send_json_error(f"Unknown experiment: {exp_name}", 400)
            return

        try:
            if noise > 0.0:
                from qex.backend import NoisyCirqBackend
                backend: CirqBackend = NoisyCirqBackend(noise)  # type: ignore[assignment]
            else:
                backend = CirqBackend()
            runner = Runner(backend, base_dir=_db_path.parent)
            record = runner.run(exp, params, config)

            # Save to store
            store = ResultStore(_db_path)
            store.save_run(record)
            store.close()
        except Exception as e:
            self.send_json_error(f"Simulation/Database write error: {str(e)}", 500)
            return

        self.send_json({
            "success": True,
            "record": {
                "run_id": record.run_id,
                "experiment_name": record.experiment_name,
                "timestamp": record.timestamp
            }
        })

    def send_json(self, data: Any, status: int = 200) -> None:
        """Helper to send JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json_error(self, message: str, status: int = 400) -> None:
        """Helper to send JSON error response."""
        self.send_json({"success": False, "error": message}, status)

    def api_compare_runs(self) -> None:
        """Calculate state fidelity between two runs."""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        run_ids1 = params.get("run_id1")
        run_ids2 = params.get("run_id2")
        
        if not run_ids1 or not run_ids2:
            self.send_json_error("Both run_id1 and run_id2 are required parameters", 400)
            return
            
        run_id1 = run_ids1[0]
        run_id2 = run_ids2[0]
        
        try:
            store = ResultStore(_db_path)
            r1 = store.get_run(run_id1)
            r2 = store.get_run(run_id2)
            store.close()
        except Exception as e:
            self.send_json_error(f"Database error: {str(e)}", 500)
            return
            
        if not r1 or not r2:
            self.send_json_error("One or both runs were not found in the database", 404)
            return
            
        try:
            rho1 = r1.get_density_matrix()
            rho2 = r2.get_density_matrix()
            
            from qex.bloch import state_fidelity
            fidelity = state_fidelity(rho1, rho2)
        except Exception as e:
            self.send_json_error(f"Fidelity calculation failed: {str(e)}", 500)
            return
            
        self.send_json({
            "success": True,
            "fidelity": fidelity,
            "run1": {
                "run_id": r1.run_id,
                "experiment_name": r1.experiment_name,
                "timestamp": r1.timestamp
            },
            "run2": {
                "run_id": r2.run_id,
                "experiment_name": r2.experiment_name,
                "timestamp": r2.timestamp
            }
        })


# Define a customized ThreadingTCPServer to avoid "Address already in use" errors on restart
class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def start_server(db_path: str, port: int = 8000) -> None:
    """Start the dashboard web server on localhost and open browser."""
    global _db_path
    _db_path = Path(db_path).resolve()

    # Ensure parent directories exist
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    server_address = ("127.0.0.1", port)
    try:
        httpd = ThreadedHTTPServer(server_address, DashboardHandler)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"qex Dashboard active at: http://127.0.0.1:{port}/")
    print("Press Ctrl+C to terminate.")

    # Automatically open dashboard in browser
    webbrowser.open(f"http://127.0.0.1:{port}/")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down qex server.")
        httpd.server_close()
        sys.exit(0)
