from torch import nn
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from pathlib import Path

DEFAULT_PATH_COVARIANCE = Path(
    "/n/home11/ccuestalazaro/sunbird/data/full_ap/covariance/ds_cross_xi_smu_zsplit_Rs20_landyszalay_randomsX50.npy",
)


class GaussianLoglike(nn.Module):
    def __init__(self, covariance):
        super().__init__()
        self.covariance = covariance

    @classmethod
    def from_file(cls, path_to_cov=DEFAULT_PATH_COVARIANCE):
        data = np.load(path_to_cov, allow_pickle=True).item()
        data = data["multipoles"][:, 0, 0]
        covariance = torch.tensor(np.cov(data.T), dtype=torch.float32)
        return cls(covariance=covariance)

    def __call__(self, inputs, targets):
        # TODO: solve this more elegantly...
        self.covariance = self.covariance.type_as(inputs)
        return (
            MultivariateNormal(
                targets,
                covariance_matrix=self.covariance,
            )
            .log_prob(inputs)
            .mean()
        )
