import torch
import torch.nn as nn
import jax
from flax import linen 



class LearnedSigmoid(nn.Module):
    def __init__(self, n_dim: int):
        super().__init__()
        initial_alpha = torch.randn(n_dim, requires_grad=True)
        initial_beta = torch.randn(n_dim, requires_grad=True)
        self.alpha = nn.Parameter(initial_alpha)
        self.beta = nn.Parameter(initial_beta)
        

    def forward(self, x):
        return (self.beta + torch.sigmoid(self.alpha * x) * (1.0 - self.beta)) * x

class FlaxLearnedSigmoid(linen.Module):
    n_dim: int

    def setup(self):
        # Initialize alpha and beta as trainable parameters
        self.alpha = self.param('alpha', linen.initializers.normal(), (self.n_dim,))
        self.beta = self.param('beta', linen.initializers.normal(), (self.n_dim,))

    def __call__(self, x):
        # Apply the learned sigmoid function
        sigmoid = self.beta + jax.nn.sigmoid(self.alpha * x) * (1.0 - self.beta)
        return sigmoid * x