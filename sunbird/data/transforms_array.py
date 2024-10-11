from abc import ABC, abstractmethod
import torch
import numpy as np
import jax.numpy as jnp


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass


class LogTransform(BaseTransform):
    def transform(self, x):
        if type(x) == torch.Tensor:
            return torch.log10(x)
        elif type(x) == np.ndarray:
            return np.log10(x)

    def inverse_transform(self, x):
        return 10**x


class ArcsinhTransform(BaseTransform):
    def transform(self, x):
        if type(x) == torch.Tensor:
            return torch.asinh(x)
        elif type(x) == np.ndarray:
            return np.arcsinh(x)
        else:
            return jnp.arcsinh(x)

    def inverse_transform(self, x):
        if type(x) == torch.Tensor:
            return torch.sinh(x)
        elif type(x) == np.ndarray:
            return np.sinh(x)
        else:
            return jnp.sinh(x)
        