from torch import nn, Tensor
import torch

class WeightedL1Loss(nn.Module):
    def __init__(self, variance: Tensor):
        """ Weighted L1 loss

        Args:
            variance (Tensor): variance tensor 
        """
        super().__init__()
        self.variance = variance

    def forward(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute weighted L1 loss

        Args:
            predictions (Tensor): model predictions
            targets (Tensor): target values

        Returns:
            float: weighted L1 loss
        """
        return torch.mean(
            torch.abs(
                (predictions - targets) / self.variance
            )
        )

class WeightedMSELoss(nn.Module):
    def __init__(self, variance: Tensor):
        """ Weighted MSE loss

        Args:
            variance (Tensor): variance tensor 
        """
        super().__init__()
        self.variance = variance

    def forward(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute weighted MSE loss

        Args:
            predictions (Tensor): model predictions
            targets (Tensor): target values

        Returns:
            float: weighted mse loss
        """
        return torch.mean(
            torch.square(
                (predictions - targets) / self.variance
            )
        )

