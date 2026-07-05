"""
CLI tool for managing and running quantum experiments in qex.
"""

import sys
import webbrowser
from pathlib import Path
from typing import Dict, Any
import click
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


def get_default_store(db_path: str) -> ResultStore:
    """Get ResultStore for specified DB path."""
    return ResultStore(Path(db_path))


@click.group()
@click.option(
    "--db-path",
    default="qex_data/qex.db",
    help="Path to SQLite database file.",
)
@click.pass_context
def main(ctx: click.Context, db_path: str) -> None:
    """qex: Lightweight experiment-runner and lab notebook for quantum computing."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


@main.command(name="list")
@click.pass_context
def list_runs(ctx: click.Context) -> None:
    """List all stored runs in the database."""
    db_path = ctx.obj["db_path"]
    try:
        store = get_default_store(db_path)
        runs = store.list_runs()
        store.close()
    except Exception as e:
        click.echo(f"Error opening database: {e}", err=True)
        sys.exit(1)

    if not runs:
        click.echo("No runs found in database.")
        return

    click.echo(f"\nRuns in database ({db_path}):")
    click.echo("-" * 90)
    click.echo(f"{'Run ID':<10} | {'Experiment':<15} | {'Backend':<12} | {'Timestamp':<20} | {'Params'}")
    click.echo("-" * 90)
    for r in runs:
        import datetime
        dt = datetime.datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        params_str = str(r.params)
        click.echo(f"{r.run_id[:8]:<10} | {r.experiment_name:<15} | {r.backend_name:<12} | {dt:<20} | {params_str}")
    click.echo("-" * 90)


@main.command(name="run")
@click.argument("experiment", type=click.Choice(["x_gate", "hadamard", "ry_sweep", "bell_state"]))
@click.option("--qubits", default=None, type=int, help="Number of qubits (1 to 10).")
@click.option("--theta", default=None, type=float, help="Rotation angle (theta) for ry_sweep experiment.")
@click.option("--noise", default=0.0, type=float, help="Depolarizing noise probability parameter (p).")
@click.pass_context
def run_experiment(ctx: click.Context, experiment: str, qubits: int, theta: float, noise: float) -> None:
    """Execute a quantum experiment and save results."""
    db_path = ctx.obj["db_path"]
    params: Dict[str, Any] = {}
    
    # Resolve qubit count
    if qubits is None:
        q_count = 2 if experiment == "bell_state" else 1
    else:
        q_count = qubits
        
    if not 1 <= q_count <= 10:
        click.echo("Error: --qubits must be between 1 and 10.", err=True)
        sys.exit(1)
        
    import cirq
    config = {"qubits": [cirq.GridQubit(0, i) for i in range(q_count)]}
    
    # Resolve the experiment
    if experiment == "x_gate":
        exp = x_gate_experiment()
        params = {}
    elif experiment == "hadamard":
        exp = hadamard_experiment()
        params = {}
    elif experiment == "ry_sweep":
        exp = ry_sweep_experiment()
        # Default theta if not specified
        t_val = theta if theta is not None else np.pi / 2
        params = {"theta": t_val}
    elif experiment == "bell_state":
        exp = bell_state_experiment()
        params = {}
        if q_count < 2:
            click.echo("Error: bell_state requires at least 2 qubits.", err=True)
            sys.exit(1)
    else:
        click.echo(f"Unknown experiment: {experiment}", err=True)
        sys.exit(1)

    click.echo(f"Running experiment: {exp.name}...")
    
    if noise > 0.0:
        from qex.backend import NoisyCirqBackend
        backend: CirqBackend = NoisyCirqBackend(noise)  # type: ignore[assignment]
    else:
        backend = CirqBackend()
    # Ensure directories match database location
    db_dir = Path(db_path).parent
    runner = Runner(backend, base_dir=db_dir)
    
    record = runner.run(exp, params, config)
    rho = record.get_density_matrix()
    
    # Save to store
    store = get_default_store(db_path)
    store.save_run(record)
    store.close()
    
    click.echo(f"Success: Run complete! Saved ID: {record.run_id}")
    click.echo(f"Density Matrix:\n{rho}")
    
    # Print advice on viewing
    if len(record.artifacts) > 0:
        art_key = list(record.artifacts.keys())[0]
        click.echo(f"Artifact created: {art_key} -> {record.artifacts[art_key]}")
        click.echo(f"To visualize, run: qex --db-path {db_path} view {record.run_id}")


@main.command(name="view")
@click.argument("run_id")
@click.pass_context
def view_run(ctx: click.Context, run_id: str) -> None:
    """Open the Bloch sphere visualization for a specific run in the browser."""
    db_path = ctx.obj["db_path"]
    store = get_default_store(db_path)
    
    # Try exact lookup or prefix match
    record = store.get_run(run_id)
    if not record:
        # Prefix lookup
        all_runs = store.list_runs()
        matches = [r for r in all_runs if r.run_id.startswith(run_id)]
        if len(matches) == 1:
            record = matches[0]
        elif len(matches) > 1:
            click.echo(f"Multiple runs match prefix '{run_id}':", err=True)
            for m in matches:
                click.echo(f"  {m.run_id}", err=True)
            store.close()
            sys.exit(1)
            
    if not record:
        click.echo(f"Run ID '{run_id}' not found in database.", err=True)
        store.close()
        sys.exit(1)
        
    store.close()
    
    # Open Bloch sphere visualization
    html_path = None
    for key, path_val in record.artifacts.items():
        if "bloch_sphere" in key:
            html_path = record._base_dir / path_val if record._base_dir else Path(path_val)
            break
            
    if not html_path or not html_path.exists():
        click.echo(f"No visualization artifact found for run ID: {record.run_id}", err=True)
        sys.exit(1)
        
    click.echo(f"Opening visualization in browser: {html_path.resolve()}")
    webbrowser.open(f"file:///{html_path.resolve().as_posix()}")


@main.command(name="ui")
@click.option("--port", default=8000, help="Port to run the UI server on.")
@click.pass_context
def run_ui(ctx: click.Context, port: int) -> None:
    """Start the interactive web dashboard server."""
    db_path = ctx.obj["db_path"]
    click.echo(f"Starting qex Web Dashboard on port {port}...")
    click.echo(f"Connecting to database: {db_path}")
    
    # Import and start ui server (implemented in Phase 3)
    from qex.ui import start_server
    start_server(db_path=db_path, port=port)


if __name__ == "__main__":
    main()
