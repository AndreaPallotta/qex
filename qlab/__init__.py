"""
qlab: A lightweight experiment-runner and lab notebook for quantum computing.

Built on top of Cirq, focused on experiments, runs, reproducibility, and visualization.
"""

from qlab.experiment import Experiment
from qlab.backend import Backend, CirqBackend
from qlab.runner import Runner
from qlab.store import ResultStore, RunRecord

__version__ = "0.1.0"
__all__ = [
    "Experiment",
    "Backend",
    "CirqBackend",
    "Runner",
    "ResultStore",
    "RunRecord",
]
