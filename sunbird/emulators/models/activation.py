import torch
import torch.nn as nn


class LearnedSigmoid(nn.Module):
    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        

    def forward(self, x):
        if self.alpha is None or self.beta is None:
            initial_alpha = torch.randn(x.shape[1:], requires_grad=True, device=x.device)
            initial_beta = torch.randn(x.shape[1:], requires_grad=True, device=x.device)
            self.alpha = nn.Parameter(initial_alpha)
            self.beta = nn.Parameter(initial_beta)
        return (self.beta + torch.sigmoid(self.alpha * x) * (1.0 - self.beta)) * x
