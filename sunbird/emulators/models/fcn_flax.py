from typing import Tuple, Optional, List, Type, Sequence, Dict
import jax.numpy as jnp
import flax.linen as nn


class FlaxFCN(nn.Module):
    """ Simple fully connected flax version of the emulator
    """
    n_input: int
    n_hidden: Sequence[int]
    act_fn: str
    n_output: int

    def setup(self,):
        """ Setup the activation function from strings
        """
        self.actvation_fn= getattr(nn, self.act_fn.lower())

    @nn.compact
    def __call__(self, x: jnp.array)->jnp.array:
        """ forward pass

        Args:
            x (jnp.array): inputs 

        Returns:
            jnp.array: outputs 
        """
        for i, dims in enumerate(self.n_hidden):
            x = nn.Dense(dims)(x)
            x = self.actvation_fn(x)
        return nn.Dense(self.n_output)(x)

    def convert_from_pytorch(self, pt_state: Dict) -> Dict:
        """ Convert the state dict from pytorch to flax

        Args:
            pt_state (Dict): state dictionary with model weights

        Returns:
            Dict: flax weights 
        """
        jax_state = dict(pt_state)
        for key, tensor in pt_state.items():
            if "mlp" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                key = key.replace("mlp.mlp", f"Dense_")
                jax_state[key] = tensor.T
        return jax_state