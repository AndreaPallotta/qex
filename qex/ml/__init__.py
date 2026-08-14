"""
qex.ml: Quantum Machine Learning extension for qex.

Optional subpackage providing Variational Quantum Eigensolver (VQE), Quantum Neural Networks (QNN),
and Neural State Tomography.
"""

from typing import Any


def has_ml() -> bool:
    # Returns True if optional ML dependencies (torch) are installed.
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _require_ml(feature_name: str) -> None:
    # Raises clear error if ML dependencies are missing.
    if not has_ml():
        raise ImportError(
            f"Feature '{feature_name}' requires PyTorch. Install optional dependencies via: pip install qex[ml]"
        )


__all__ = ["has_ml", "VQE", "QNN", "QAOA", "NeuralTomography", "QuantumOptimizer"]


def __getattr__(name: str) -> Any:
    # Lazy exports for ML modules.
    if name == "VQE":
        _require_ml("VQE")
        from qex.ml.vqe import VQE
        return VQE
    elif name == "QNN":
        _require_ml("QNN")
        from qex.ml.qnn import QNN
        return QNN
    elif name == "QAOA":
        _require_ml("QAOA")
        from qex.ml.qaoa import QAOA
        return QAOA
    elif name == "NeuralTomography":
        _require_ml("NeuralTomography")
        from qex.ml.tomography import NeuralTomography
        return NeuralTomography
    elif name == "QuantumOptimizer":
        _require_ml("QuantumOptimizer")
        from qex.ml.optimizer import QuantumOptimizer
        return QuantumOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
