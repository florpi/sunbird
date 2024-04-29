import torch
import torch.nn as nn


class LearnedSigmoid(nn.Module):
    def __init__(self, n_dim: int):
        super().__init__()
        initial_alpha = torch.randn(n_dim, requires_grad=True)
        initial_beta = torch.randn(n_dim, requires_grad=True)
        self.alpha = nn.Parameter(initial_alpha)
        self.beta = nn.Parameter(initial_beta)
        

    def forward(self, x):
        return (self.beta + torch.sigmoid(self.alpha * x) * (1.0 - self.beta)) * x
