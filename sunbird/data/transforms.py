from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
import xarray as xr
import numpy as np
import sys
import pickle


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        return

    @abstractmethod
    def inverse_transform(self, summary: xr.DataArray, errors) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        return

    def get_parameter_dict(
        self,
    ) -> Dict:
        """get parameters needed for transform

        Returns:
            Dict: dictionary of parameteres
        """
        return self.__dict__

    def fit_transform(
        self,
        summary: xr.DataArray,
    ) -> xr.DataArray:
        """Fit the transform from data in summary and transform summary.


        Args:
            summary (xr.DataArray): data to fit and transform
            dimensions (xr.DataArray): dimensions over which to fit

        Returns:
            xr.DataArray: transformed summary
        """
        self.fit(
            summary,
        )
        return self.transform(summary)


class Transforms:
    def __init__(self, transforms: List[BaseTransform]):
        """Combine multiple transforms into one

        Args:
            transforms (List[BaseTransform]): list of transforms to combine
        """
        self.transforms = transforms

    @classmethod
    def from_file(cls, filename: str) -> "Transforms":
        """Load transforms from file

        Args:
            filename (str): file to load

        Returns:
            Transforms: Transforms object
        """
        with open(filename, "rb") as f:
            param_dict = pickle.load(f)
        transforms = []
        for key, value in param_dict.items():
            transforms.append(getattr(sys.modules[__name__], key)(**value))
        return cls(
            transforms=transforms,
        )

    def fit_transform(
        self,
        summary,
        path_to_store=None,
    ):
        """Fit the transform from data in summary and transform summary.


        Args:
            summary (xr.DataArray): data to fit and transform
            path_to_store (str, optional): path to store parameters. Defaults to None.

        Returns:
            xr.DataArray: transformed summary
        """
        for transform in self.transforms:
            if hasattr(transform, "fit"):
                summary = transform.fit_transform(
                    summary,
                )
            else:
                summary = transform.transform(summary)
        if path_to_store is not None:
            self.store_transform_params(path_to_store=path_to_store)
        return summary

    def store_transform_params(self, path_to_store: str):
        """
        Store the parameters of the transforms

        Args:
            path_to_store (str): path to store parameters
        """
        param_dict = {}
        for transform in self.transforms:
            if hasattr(transform, "fit"):
                param_dict[
                    transform.__class__.__name__
                ] = transform.get_parameter_dict()
            else:
                param_dict[transform.__class__.__name__] = {}
        with open(path_to_store, "wb") as f:
            pickle.dump(param_dict, f)

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        summary = summary.copy()
        for transform in self.transforms:
            summary = transform.transform(summary)
        return summary

    def inverse_transform(
        self,
        summary: xr.DataArray,
        errors: xr.DataArray,
        summary_dimensions: List[str] = None,
        batch: bool = False,
    ) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        if type(summary) is torch.Tensor:
            summary = summary.detach().clone()
        else:
            summary = summary.copy()
        if type(errors) is torch.Tensor:
            errors = errors.clone()
        else:
            errors = errors.copy()
        for transform in self.transforms[::-1]:
            summary, errors = transform.inverse_transform(
                summary, errors, summary_dimensions=summary_dimensions, batch=batch
            )
        return summary, errors


class Normalize(BaseTransform):
    def __init__(
        self,
        training_min=None,
        training_max=None,
        dimensions: Optional[List[str]] = None,
        **kwargs,
    ):
        """Normalize the summary statistics

        Args:
            training_min (float, optional): minimum value for training. Defaults to None.
            training_max (float, optional): maximum value for training. Defaults to None.
            dimensions (List[str], optional): dimensions over which to normalize. Defaults to None.
        """
        self.training_min = training_min
        self.training_max = training_max
        self.dimensions = dimensions

    def fit(
        self,
        summary: xr.DataArray,
    ):
        if type(summary) is np.ndarray:
            self.training_min = summary.min(axis=self.dimensions)
            self.training_max = summary.max(axis=self.dimensions)
        else:
            self.training_min = summary.min(dim=self.dimensions)
            self.training_max = summary.max(dim=self.dimensions)

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        return (summary - self.training_min) / (self.training_max - self.training_min)

    def inverse_transform(
        self,
        summary: xr.DataArray,
        errors,
        summary_dimensions: List[str] = None,
        batch=False,
    ) -> xr.DataArray:
        if type(summary) is xr.DataArray:
            inv_summary = (
                summary * (self.training_max - self.training_min) + self.training_min
            )
            inv_errors = errors * (self.training_max - self.training_min)
        else:
            training_min = self.training_min.values
            training_max = self.training_max.values
            if summary_dimensions is not None:
                avg_dims = [
                    summary_dimensions.index(dim)
                    for dim in self.dimensions
                    if dim in summary_dimensions
                ]
                training_min = np.expand_dims(training_min, axis=avg_dims)
                training_max = np.expand_dims(training_max, axis=avg_dims)
            if batch:
                training_min = training_min[np.newaxis, ...]
                training_max = training_max[np.newaxis, ...]
            inv_summary = summary * (training_max - training_min) + training_min
            inv_errors = errors * (training_max - training_min)
        return inv_summary, inv_errors


class Standarize(BaseTransform):
    def __init__(
        self,
        training_mean=None,
        training_std=None,
        dimensions: Optional[List[str]] = None,
        **kwargs,
    ):
        """Normalize the summary statistics

        Args:
            training_min (float, optional): minimum value for training. Defaults to None.
            training_max (float, optional): maximum value for training. Defaults to None.
        """
        self.training_mean = training_mean
        self.training_std = training_std
        self.dimensions = dimensions

    def fit(
        self,
        summary,
    ):
        if type(summary) is np.ndarray:
            self.training_mean = summary.mean(axis=self.dimensions)
            self.training_std = summary.std(axis=self.dimensions)
        else:
            self.training_mean = summary.mean(dim=self.dimensions)
            self.training_std = summary.std(dim=self.dimensions)

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        return (summary - self.training_mean) / self.training_std

    def inverse_transform(
        self,
        summary: xr.DataArray,
        errors: xr.DataArray,
        summary_dimensions: List[str] = None,
        batch: bool = False,
    ) -> xr.DataArray:
        if type(summary) is xr.DataArray:
            inv_summary = summary * self.training_std + self.training_mean
            inv_errors = errors * self.training_std
        else:
            training_mean = self.training_mean.values
            training_std = self.training_std.values
            if summary_dimensions is not None:
                avg_dims = [
                    summary_dimensions.index(dim)
                    for dim in self.dimensions
                    if dim in summary_dimensions
                ]
                training_mean = np.expand_dims(training_mean, axis=avg_dims)
                training_std = np.expand_dims(training_std, axis=avg_dims)
            if batch:
                training_mean = training_mean[np.newaxis, ...]
                training_std = training_std[np.newaxis, ...]
            inv_summary = summary * training_std + training_mean
            inv_errors = errors * training_std
        return inv_summary, inv_errors


class Log(BaseTransform):
    def __init__(
        self,
        min_value=None,
        **kwargs,
    ):
        """Transform to log

        Args:
            min_value (float, optional): minimum value of the statistic. Defaults to 0.011.
        """
        self.min_value = min_value

    def fit(self, summary: xr.DataArray):
        """Fit the transform

        Args:
            summary (xr.DataArray): summary to fit the transform to
        """
        self.min_value = 1.01 * summary.min()
        if self.min_value == 0.0:
            self.min_value = -0.01

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        summary = np.log10(summary - self.min_value)
        return summary

    def inverse_transform(
        self,
        summary: xr.DataArray,
        errors: xr.DataArray,
    ) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        inv_summary = 10**summary + self.min_value
        # TODO: how to transform errors?
        return inv_summary, errors


class S2(BaseTransform):
    def __init__(
        self,
        **kwargs,
    ):
        pass

    def fit(self, summary: xr.DataArray):
        self.s = summary.s

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        return summary * self.s**2

    def inverse_transform(
        self, summary: xr.DataArray, errors: xr.DataArray
    ) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        summary /= self.s**2
        errors /= self.s**2
        return summary, errors
