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
    
    @abstractmethod
    def get_jacobian_diagonal(self, y):
        """
        Get the diagonal of the Jacobian matrix df/dy for transforming covariance matrices.
        
        For an element-wise transformation f(y), the transformed covariance is:
        Cov_transformed = diag(J) @ Cov @ diag(J)
        where J = df/dy is the Jacobian diagonal.
        
        Parameters
        ----------
        y : array_like
            Data vector in the original (untransformed) space.
            
        Returns
        -------
        array_like
            Diagonal of the Jacobian matrix, same shape as y.
        """
        pass


class LogTransform(BaseTransform):
    def transform(self, x):
        if type(x) == torch.Tensor:
            return torch.log10(x)
        elif type(x) == np.ndarray:
            return np.log10(x)

    def inverse_transform(self, x):
        return 10**x
    
    def get_jacobian_diagonal(self, y):
        """
        Get Jacobian diagonal for log10 transform: d(log10(y))/dy = 1/(y * ln(10))
        
        Parameters
        ----------
        y : array_like
            Data vector in the original (untransformed) space.
            
        Returns
        -------
        array_like
            Jacobian diagonal: 1/(y * ln(10))
        """
        if type(y) == torch.Tensor:
            return 1.0 / (y * torch.log(torch.tensor(10.0)))
        elif type(y) == np.ndarray:
            return 1.0 / (y * np.log(10.0))
        else:
            return 1.0 / (y * jnp.log(10.0))


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
    
    def get_jacobian_diagonal(self, y):
        """
        Get Jacobian diagonal for arcsinh transform: d(arcsinh(y))/dy = 1/sqrt(1 + y^2)
        
        Parameters
        ----------
        y : array_like
            Data vector in the original (untransformed) space.
            
        Returns
        -------
        array_like
            Jacobian diagonal: 1/sqrt(1 + y^2)
        """
        if type(y) == torch.Tensor:
            return 1.0 / torch.sqrt(1.0 + y**2)
        elif type(y) == np.ndarray:
            return 1.0 / np.sqrt(1.0 + y**2)
        else:
            return 1.0 / jnp.sqrt(1.0 + y**2)

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
    
    def get_jacobian_diagonal(self, y):
        """
        Get Jacobian diagonal for affine transform: d(y * std + mean)/dy = std
        
        Parameters
        ----------
        y : array_like
            Data vector in the original (untransformed) space.
            
        Returns
        -------
        array_like
            Jacobian diagonal: std (broadcast to match y shape)
        """
        if type(y) == torch.Tensor:
            return torch.ones_like(y) * self.std
        elif type(y) == np.ndarray:
            return np.ones_like(y) * self.std.numpy()
        else:
            return jnp.ones_like(y) * self.std.numpy()

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
    
    def get_jacobian_diagonal(self, y):
        """
        Get Jacobian diagonal for standardization: d((y - mean) / std)/dy = 1/std
        
        Parameters
        ----------
        y : array_like
            Data vector in the original (untransformed) space.
            
        Returns
        -------
        array_like
            Jacobian diagonal: 1/std (broadcast to match y shape)
        """
        if type(y) == torch.Tensor:
            return torch.ones_like(y) / self.std
        elif type(y) == np.ndarray:
            return np.ones_like(y) / self.std.numpy()
        else:
            return jnp.ones_like(y) / self.std.numpy()
        