from abc import ABC, abstractmethod
import xarray as xr
import numpy as np


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
    def inverse_transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        return


class LogSqrt(BaseTransform):
    def __init__(
        self,
        min_monopole: float = 0.011,
        min_quadrupole: float = -30.0,
    ):
        """Transform the monopole and quadrupole to log and sqrt, respectively

        Args:
            min_monopole (float, optional): minimum monopole value. Defaults to 0.011.
            min_quadrupole (float, optional): minimum quadrupole value. Defaults to -30..
        """
        self.min_monopole = min_monopole
        self.min_quadrupole = min_quadrupole

    def transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Transform a summary

        Args:
            summary (xr.DataArray): summary to transform

        Returns:
            xr.DataArray: transformed summary
        """
        summary.loc[{"multipoles": 0}] = np.log10(
            summary.sel(multipoles=0) - self.min_monopole
        )
        summary.loc[{"multipoles": 1}] = (
            summary.sel(multipoles=1) - self.min_quadrupole
        ) ** 0.5
        return summary

    def inverse_transform(self, summary: xr.DataArray) -> xr.DataArray:
        """Inverse the transform

        Args:
            summary (xr.DataArray): transformed summary

        Returns:
            xr.DataArray: original summary
        """
        summary.loc[{"multipoles": 0}] = 10 ** (
            summary.sel(multipoles=0) + self.min_monopole
        )
        summary.loc[{"multipoles": 1}] = (
            summary.sel(multipoles=1) + self.min_quadrupole
        ) ** 2
        return summary
