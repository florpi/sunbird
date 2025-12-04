import torch
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod

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

class WeiLiuOutputTransForm(BaseTransform):
    """Class to reconcile output the Minkowski functionals model
    trained with Wei Liu's scripts with those from the ACM repository.
    """
    def __init__(self,):
        # self.data_dict = np.load('/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/minkowski_dummy.npy', allow_pickle=True).item()
        self.data_dict = np.load('/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/Minkowski_Combine_4Rgs_85cos_lhc.npy', allow_pickle=True).item()
        self.mean = torch.tensor(self.data_dict['train_y_mean'],dtype=torch.float32)
        self.std = torch.tensor(self.data_dict['train_y_std'],dtype=torch.float32)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x * self.std + self.mean

class WeiLiuInputTransform(BaseTransform):
    """Class to reconcile input of the Minkowski functionals model
    trained with Wei Liu's scripts with those from the ACM repository.
    """
    def __init__(self,):
        # self.data_dict = np.load('/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/minkowski_dummy.npy', allow_pickle=True).item()
        self.data_dict = np.load('/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/Minkowski_Combine_4Rgs_85cos_lhc.npy', allow_pickle=True).item()
        self.mean = torch.tensor(self.data_dict['train_x_mean'],dtype=torch.float32)
        self.std = torch.tensor(self.data_dict['train_x_std'],dtype=torch.float32)

    def transform(self, x):
        return ((x - self.mean) / self.std).to(torch.float32)

    def inverse_transform(self, x):
        return x
        