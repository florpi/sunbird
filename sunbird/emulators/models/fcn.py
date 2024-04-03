from typing import OrderedDict, Dict, List
import numpy as np
from torch import nn, Tensor
import torch

from sunbird.emulators.models import BaseModel
from sunbird.covariance import CovarianceMatrix
from sunbird.emulators.loss import MultivariateGaussianNLLLoss, get_cholesky_decomposition_covariance, WeightedL1Loss, WeightedMSELoss


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class FCN(BaseModel):
    def __init__(self, *args, **kwargs):
        """Fully connected neural network with variable architecture"""
        super().__init__()
        self.n_input = kwargs["n_input"]
        self.n_output = kwargs["n_output"]
        self.n_hidden = kwargs["n_hidden"]
        dropout_rate = kwargs["dropout_rate"]
        self.learning_rate = kwargs["learning_rate"]
        self.weight_decay = kwargs["weight_decay"]
        self.act_fn_str = kwargs["act_fn"]
        act_fn = getattr(nn, self.act_fn_str)()
        self.loss = kwargs["loss"]
        self.data_dim = self.n_output
        if self.loss == "learned_gaussian":
            self.n_output *= 2
        elif self.loss == "multivariate_learned_gaussian":
            self.n_output += (self.n_output*(self.n_output+1))//2
        self.mlp = self.get_model(
            n_input=self.n_input,
            n_hidden=self.n_hidden,
            act_fn=act_fn,
            n_output=self.n_output,
            dropout_rate=dropout_rate,
        )
        if kwargs["load_loss"]:
            self.load_loss(**kwargs)
            self.save_hyperparameters(
                ignore=[
                    "output_transforms",
                    "input_transforms",
                    "select_filters",
                    "slice_filters",
                ],
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Model arguments that could vary

        Args:
            parent_parser (parser): parser

        Returns:
            parser: updated parser
        """
        parser = parent_parser.add_argument_group("FCN")
        parser.add_argument("--act_fn", type=str, default="SiLU")
        parser.add_argument(
            "--n_hidden",
            action="store",
            type=int,
            default=[100, 100],
            nargs="+",
        )
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--loss", type=str, default="mae")
        parser.add_argument("--load_loss", type=bool, default=True)
        return parent_parser

    @classmethod
    def from_argparse_args(
        cls,
        args: Dict,
    ) -> "FCN":
        """Create neural net from parsed args

        Args:
            args (args): command line arguments

        Returns:
            FCN: fully connected neural network
        """
        if type(args) is not dict:
            vargs = vars(args)
        else:
            vargs = args
        select_filters, slice_filters = {}, {}
        for key, value in vargs.items():
            if "select" in key:
                key_to_filter = key.split("_")[-1]
                if key_to_filter != "gpu":
                    select_filters[key_to_filter] = value
            elif "slice" in key:
                slice_filters[key.split("_")[-1]] = value
        return cls(
            select_filters=select_filters,
            slice_filters=slice_filters,
            **vargs,
        )
    
    @property
    def flax_attributes(self,):
        return {'n_input': self.n_input,
                'n_hidden': self.n_hidden,
                'act_fn': self.act_fn_str,
                'n_output': self.n_output,
                'predict_errors': True if self.loss == "learned_gaussian" else False,
        }

    def get_model(
        self,
        n_input: int,
        n_hidden: List[int],
        act_fn: str,
        n_output: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        """Get mlp model

        Args:
            n_input (int): dimensionality input
            n_hidden (List[int]): number of hidden units per layer
            act_fn (str): activation function
            n_output (int): number of outputs
            dropout_rate (float): dropout rate

        Returns:
            nn.Sequential: model
        """
        dropout = nn.Dropout(dropout_rate)
        model = []
        for layer in range(len(n_hidden)):
            n_left = n_input if layer == 0 else n_hidden[layer - 1]
            model.append((f"mlp{layer}", nn.Linear(n_left, n_hidden[layer])))
            model.append((f"act{layer}", act_fn))
            model.append((f"dropout{layer}", dropout))
        model.append((f"mlp{layer+1}", nn.Linear(n_hidden[layer], n_output)))
        return nn.Sequential(OrderedDict(model))

    def load_loss(self, loss: str, **kwargs):
        """Load loss function

        Args:
            loss (str): loss to load
        """
        if "weighted" in loss:
            covariance = CovarianceMatrix(
                statistics=[kwargs["statistic"]],
                slice_filters=kwargs.get("slice_filters", None),
                select_filters=kwargs.get("select_filters", None),
                output_transforms={kwargs["statistic"]: kwargs["output_transforms"]},
                dataset=kwargs["abacus_dataset"],
            ).get_covariance_data(
                volume_scaling=64.0,
            )
            covariance = Tensor(
                covariance.astype(np.float32),
            )
            if loss == "weighted_mae":
                self.loss_fct = WeightedL1Loss(
                    variance=torch.sqrt(torch.diagonal(covariance))
                )
            elif loss == "weighted_mse":
                self.loss_fct = WeightedMSELoss(variance=torch.diagonal(covariance))
        elif loss == "learned_gaussian":
            self.loss_fct = nn.GaussianNLLLoss()
        elif loss == "multivariate_learned_gaussian":
            self.loss_fct = MultivariateGaussianNLLLoss()
        elif loss == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss == "mae":
            self.loss_fct = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {loss} not implemented")

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward model

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output
        """
        if self.loss == "learned_gaussian":
            y_pred = self.mlp(x)
            y_pred, y_var = torch.chunk(y_pred, 2, dim=-1)
            y_var = nn.Softplus()(y_var)
            return y_pred, y_var
        elif self.loss == "multivariate_learned_gaussian":
            y_pred = self.mlp(x)
            y_cov = y_pred[..., self.data_dim:]
            y_pred = y_pred[..., :self.data_dim]
            L = get_cholesky_decomposition_covariance(y_cov, data_dim=self.data_dim,)
            return y_pred, L 
        y_pred = self.mlp(x)
        y_var = torch.zeros_like(y_pred)
        return y_pred, y_var

    def _compute_loss(self, batch, batch_idx) -> float:
        """Compute loss in batch

        Args:
            batch: batch with x and y
            batch_idx: batch idx

        Returns:
            float: loss
        """
        x, y = batch
        y_pred, y_var = self.forward(x)
        if self.loss == "learned_gaussian":
            return self.loss_fct(y_pred, y, y_var)
        elif self.loss == "multivariate_learned_gaussian":
            return self.loss_fct(y_pred, y, y_var)
        return self.loss_fct(y, y_pred)

