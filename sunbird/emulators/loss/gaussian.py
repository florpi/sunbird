import torch
from torch import nn, Tensor

def get_cholesky_decomposition_covariance(values: Tensor, data_dim: int)->Tensor:
    """ Starting from a vector of values, return the lower triangular matrix L such that LL^T = covariance
    note that values will have data_dim*(data_dim+1)/2 elements (the lower triangular part of the covariance matrix

    Args:
        values (Tensor): lower triangular part of the covariance matrix 
        data_dim (int): dimension of the data 

    Returns:
        Tensor: lower triangular matrix L such that LL^T = covariance 
    """
    if values.dim() == 1: 
        values = values.unsqueeze(0) 
    batch_size = values.size(0)
    mask = torch.tril(torch.ones(1, data_dim, data_dim, dtype=bool)).to(values.device)
    mask = mask.repeat(batch_size, 1, 1) # Repeat the mask for the batch size
    L_batch = torch.zeros_like(mask, dtype=values.dtype)
    values_flat = values.reshape(-1)
    L_batch[mask] = values_flat
    # ensure that each matrix in the batch of L has a positive diagonal:
    diagonal_indices = torch.arange(L_batch.size(1), device=L_batch.device)
    L_batch[:, diagonal_indices, diagonal_indices] = torch.exp(L_batch[:, diagonal_indices, diagonal_indices])
    return L_batch

class MultivariateGaussianNLLLoss(nn.Module):
    def __init__(self):
        super(MultivariateGaussianNLLLoss, self).__init__()

    def __call__(self, predictions: Tensor, targets: Tensor, L: Tensor, epsilon=1.e-5) -> float:
        """Given a set of inputs and targets, estimate the negative log-likelihood of the predicted values

        Args:
            predictions (Tensor): model predictions
            targets (Tensor): target values
            L: (Tensor) tensor representing the lower triangular matrices


        Returns:
            float: log-likelihood of the targets 
        """
        k = targets.size(-1)
        covariance = torch.einsum('bij,bjk->bik', L, L.transpose(-2, -1))
        # regularize the covariance matrix
        covariance += torch.eye(k, device=covariance.device) * epsilon
        diff = (targets - predictions).unsqueeze(-1)
        inv_covariance = torch.linalg.inv(covariance)
        log_det_covariance = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1) # log determinant of LL^T = 2 * log determinant of 
        loss = 0.5 * (torch.einsum('bik,bik->b', diff, torch.einsum('bij,bjk->bik', inv_covariance, diff)) 
                      + log_det_covariance 
                      + k * torch.log(torch.tensor(2.) * torch.pi))
        return loss.mean()


class GaussianNLoglike(nn.Module):
    def __init__(self, covariance: Tensor):
        """Class to compute the negative Gaussian log-likelihood given a covariance matrix

        Args:
            covariance (Tensor): covariance matrix
        """
        super().__init__()
        self.covariance = covariance
        self.inverse_covariance = torch.linalg.inv(self.covariance).to(covariance.device)

    def __call__(self, predictions: Tensor, targets: Tensor) -> float:
        """Given a set of inputs and targets, estimate the negative log-likelihood of the predicted values

        Args:
            predictions (Tensor): model predictions
            targets (Tensor): target values

        Returns:
            float: log-likelihood of the predicitons
        """
        diff = (predictions - targets).to(self.covariance.device)
        right = torch.einsum("ij,kj->ki", self.inverse_covariance, diff)
        return 0.5 * (
            torch.einsum(
                "...j,...j", diff, right
            )
        ).mean()
