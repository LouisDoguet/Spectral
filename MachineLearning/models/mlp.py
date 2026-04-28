"""MLP architecture for artificial viscosity prediction."""
import torch
import torch.nn as nn


class ArtificialViscosityNet(nn.Module):
    """
    Input  : (batch, 3 * n_total)  — [rho | rhou | e] concatenated
    Output : (batch, n_total)      — eps per node, guaranteed >= 0
    """

    def __init__(self, n_total: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_total, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_total),
            nn.Softplus(),  # enforces eps >= 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
