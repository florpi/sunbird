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


class HighKTaperTransform(BaseTransform):
    """Apply a fixed high-k taper across flattened power-spectrum features."""

    def __init__(self, feature_k, k0=0.1, slope=0.75):
        self.feature_k = np.asarray(feature_k, dtype=float)
        if self.feature_k.ndim != 1:
            raise ValueError("feature_k must be a one-dimensional array.")
        if self.feature_k.size == 0:
            raise ValueError("feature_k must contain at least one value.")
        if k0 <= 0:
            raise ValueError("k0 must be positive.")
        self.k0 = float(k0)
        self.slope = float(slope)
        self.weights = np.where(
            self.feature_k < self.k0,
            1.0,
            (self.feature_k / self.k0) ** (-self.slope),
        )

    def _check_feature_shape(self, x):
        if x.shape[-1] != self.weights.size:
            raise ValueError(
                "Input last dimension must match feature_k length "
                f"({x.shape[-1]} != {self.weights.size})."
            )

    def _numpy_weights(self, x):
        self._check_feature_shape(x)
        shape = (1,) * (x.ndim - 1) + (self.weights.size,)
        return self.weights.astype(x.dtype, copy=False).reshape(shape)

    def _torch_weights(self, x):
        self._check_feature_shape(x)
        shape = (1,) * (x.ndim - 1) + (self.weights.size,)
        return torch.as_tensor(
            self.weights,
            dtype=x.dtype,
            device=x.device,
        ).reshape(shape)

    def _jax_weights(self, x):
        self._check_feature_shape(x)
        shape = (1,) * (x.ndim - 1) + (self.weights.size,)
        return jnp.asarray(self.weights, dtype=x.dtype).reshape(shape)

    def _weights_like(self, x):
        if type(x) == torch.Tensor:
            return self._torch_weights(x)
        if type(x) == np.ndarray:
            return self._numpy_weights(x)
        return self._jax_weights(x)

    def transform(self, x):
        return x * self._weights_like(x)

    def inverse_transform(self, x):
        return x / self._weights_like(x)

    def get_jacobian_diagonal(self, y):
        return self._weights_like(y) * (y * 0 + 1)


class FiducialSpectrumNormalizeTransform(BaseTransform):
    """Normalize flattened spectrum features by a fixed fiducial divisor."""

    def __init__(self, divisor):
        self.divisor = np.asarray(divisor, dtype=float)
        if self.divisor.ndim != 1:
            raise ValueError("divisor must be a one-dimensional array.")
        if self.divisor.size == 0:
            raise ValueError("divisor must contain at least one value.")
        if not np.isfinite(self.divisor).all() or np.any(self.divisor == 0):
            raise ValueError("divisor must contain finite non-zero values.")

    def _check_feature_shape(self, x):
        if x.shape[-1] != self.divisor.size:
            raise ValueError(
                "Input last dimension must match divisor length "
                f"({x.shape[-1]} != {self.divisor.size})."
            )

    def _divisor_like(self, x):
        self._check_feature_shape(x)
        shape = (1,) * (x.ndim - 1) + (self.divisor.size,)
        if type(x) == torch.Tensor:
            return torch.as_tensor(self.divisor, dtype=x.dtype, device=x.device).reshape(shape)
        if type(x) == np.ndarray:
            return self.divisor.astype(x.dtype, copy=False).reshape(shape)
        return jnp.asarray(self.divisor, dtype=x.dtype).reshape(shape)

    def transform(self, x):
        return x / self._divisor_like(x)

    def inverse_transform(self, x):
        return x * self._divisor_like(x)

    def get_jacobian_diagonal(self, y):
        return (1.0 / self._divisor_like(y)) * (y * 0 + 1)


class ArrayTransformSequence(BaseTransform):
    """Compose array transforms into one checkpointable transform."""

    def __init__(self, transforms):
        self.transforms = list(transforms)
        if not self.transforms:
            raise ValueError("transforms must contain at least one transform.")

    def transform(self, x):
        for transform in self.transforms:
            x = transform.transform(x)
        return x

    def inverse_transform(self, x):
        for transform in reversed(self.transforms):
            x = transform.inverse_transform(x)
        return x

    def get_jacobian_diagonal(self, y):
        diagonal = y * 0 + 1
        current = y
        for transform in self.transforms:
            diagonal = diagonal * transform.get_jacobian_diagonal(current)
            current = transform.transform(current)
        return diagonal

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
        
