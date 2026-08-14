import numpy as np
from typing import Dict, Any
from qex.ml import _require_ml

_require_ml("NeuralTomography")
import torch


class NeuralTomography:
    # Reconstructs clean quantum density matrices from noisy measurement samples.

    def __init__(self, num_qubits: int = 2):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits

    def denoise_density_matrix(self, noisy_rho: np.ndarray, iterations: int = 30) -> Dict[str, Any]:
        # Enforces positive semi-definiteness and unit trace on noisy density matrix.
        dim = self.dim
        real_part = torch.nn.Parameter(torch.tensor(np.real(noisy_rho), dtype=torch.float32))
        imag_part = torch.nn.Parameter(torch.tensor(np.imag(noisy_rho), dtype=torch.float32))

        optimizer = torch.optim.Adam([real_part, imag_part], lr=0.01)

        for _ in range(iterations):
            optimizer.zero_grad()
            t_mat = torch.complex(real_part, imag_part)

            # Enforce Hermiticity and positivity T^\dagger T.
            rho_pred = torch.matmul(t_mat.conj().T, t_mat)
            trace_val = torch.trace(rho_pred)
            rho_norm = rho_pred / (trace_val + 1e-8)

            target = torch.complex(torch.tensor(np.real(noisy_rho), dtype=torch.float32), torch.tensor(np.imag(noisy_rho), dtype=torch.float32))
            loss = torch.norm(rho_norm - target)
            loss.backward()
            optimizer.step()

        final_t = torch.complex(real_part.detach(), imag_part.detach())
        final_rho_unscaled = torch.matmul(final_t.conj().T, final_t).numpy()
        clean_rho = final_rho_unscaled / np.trace(final_rho_unscaled)

        purity = float(np.real(np.trace(clean_rho @ clean_rho)))
        return {
            "num_qubits": self.num_qubits,
            "clean_density_matrix": clean_rho,
            "purity": purity,
            "is_valid_state": bool(np.isclose(np.trace(clean_rho), 1.0) and purity <= 1.0001),
        }
