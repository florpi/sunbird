import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Dict, Optional
from sunbird.covariance import CovarianceMatrix


class GaussianNLoglike(nn.Module):
    def __init__(self, covariance: Tensor):
        """Class to compute the negative Gaussian log-likelihood given a covariance matrix

        Args:
            covariance (Tensor): covariance matrix
        """
        super().__init__()
        self.covariance = covariance
        self.inverse_covariance = torch.linalg.inv(self.covariance)
        self.n_bins = self.covariance.shape[0]

    @classmethod
    def from_statistics(
        cls,
        statistics: List[str],
        slice_filters: Dict = None,
        select_filters: Dict = None,
        standarize_covariance: bool = False,
        normalize_covariance: bool = False,
        volume_scaling: float = 64.0,  # Scale by 64. to the 2Gpc/h volme of emulator boxes, 8 would be for cmass
        normalization_dict: Optional[Dict] = None,
    ):
        """Initialize a Gaussian log-likelihood from a list of statistics and filters
        Args:
            statistics (List[str]): list of statistics to use
            slice_filters (Dict): dictionary with slice filters on given coordinates
            select_filters (Dict): dictionary with select filters on given coordinates
        """
        covariance = CovarianceMatrix(
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
            standarize_covariance=standarize_covariance,
            normalize_covariance=normalize_covariance,
            normalization_dict=normalization_dict,
        ).get_covariance_data(
            volume_scaling=volume_scaling,
        )
        return cls(
            covariance=Tensor(
                covariance.astype(np.float32),
            )
        )

    def __call__(self, predictions: Tensor, targets: Tensor) -> float:
        """Given a set of inputs and targets, estimate the negative log-likelihood of the predicted values

        Args:
            predictions (Tensor): model predictions
            targets (Tensor): target values

        Returns:
            float: log-likelihood of the predicitons
        """
        diff = predictions - targets
        # TODO: this shouldn't be necessary, buffer has been defined?
        self.inverse_covariance = self.inverse_covariance.to(diff.device)
        right = torch.einsum("ij,kj->ki", self.inverse_covariance, diff)
        return 0.5 * (torch.einsum("...j,...j", diff, right)).mean() / self.n_bins
