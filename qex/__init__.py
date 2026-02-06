"""
qex: A lightweight experiment-runner and lab notebook for quantum computing.

Built on top of Cirq, focused on experiments, runs, reproducibility, and visualization.
"""

__version__ = "0.1.2"
__all__ = [
    "Experiment",
    "Backend",
    "CirqBackend",
    "Runner",
    "ResultStore",
    "RunRecord",
]


def __getattr__(name: str):
    """
    Lazy import to avoid slow imports of heavy dependencies (cirq, numpy).
    Heavy modules load only when specific classes are accessed.
    """
    if name == "Experiment":
        from qex.experiment import Experiment
        return Experiment
    elif name == "Backend":
        from qex.backend import Backend
        return Backend
    elif name == "CirqBackend":
        from qex.backend import CirqBackend
        return CirqBackend
    elif name == "Runner":
        from qex.runner import Runner
        return Runner
    elif name == "ResultStore":
        from qex.store import ResultStore
        return ResultStore
    elif name == "RunRecord":
        from qex.store import RunRecord
        return RunRecord
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
