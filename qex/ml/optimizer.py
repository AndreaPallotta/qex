import numpy as np
from typing import Callable, Tuple
from qex.ml import _require_ml

_require_ml("QuantumOptimizer")


class QuantumOptimizer:
    # SPSA optimizer specifically designed for noisy quantum loss landscapes.

    def __init__(self, a: float = 0.1, c: float = 0.1, alpha: float = 0.602, gamma: float = 0.101):
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma

    def spsa_step(self, cost_fn: Callable[[np.ndarray], float], params: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        # Performs a single SPSA optimization step given iteration counter k.
        ak = self.a / ((k + 1.0) ** self.alpha)
        ck = self.c / ((k + 1.0) ** self.gamma)

        delta = np.random.choice([-1.0, 1.0], size=params.shape)

        p_plus = params + ck * delta
        p_minus = params - ck * delta

        cost_plus = cost_fn(p_plus)
        cost_minus = cost_fn(p_minus)

        ghat = (cost_plus - cost_minus) / (2.0 * ck * delta)
        new_params = params - ak * ghat
        current_cost = cost_fn(new_params)

        return new_params, current_cost
