import flax.linen as nn

class SimpleNN(nn.Module):
    num_hidden_features: int = 64 
    num_output: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.tanh(x)  
        x = nn.Dense(features=self.num_output)(x)
        return x