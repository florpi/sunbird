from typing import OrderedDict, Dict
import numpy as np
from torch import nn, Tensor

from sunbird.models.models import BaseModel
from sunbird.covariance import CovarianceMatrix
from sunbird.models.loss import GaussianNLoglike


class FCN(BaseModel):
    def __init__(self, *args, **kwargs):
        """Fully connected neural network with variable architecture"""
        super().__init__()
        n_input = kwargs["n_input"]
        n_output = kwargs["n_output"]
        n_hidden = kwargs["n_hidden"]
        n_layers = len(n_hidden)
        dropout_rate = kwargs["dropout_rate"]
        self.dropout = nn.Dropout(dropout_rate)
        self.learning_rate = kwargs["learning_rate"]
        self.weight_decay = kwargs["weight_decay"]
        act_fn = getattr(nn, kwargs["act_fn"])()
        model = []
        for layer in range(n_layers):
            n_left = n_input if layer == 0 else n_hidden[layer - 1]
            model.append((f"mlp{layer}", nn.Linear(n_left, n_hidden[layer])))
            model.append((f"act{layer}", act_fn))
            model.append((f"dropout{layer}", self.dropout))
        model.append((f"mlp{layer+1}", nn.Linear(n_hidden[layer], n_output)))
        self.mlp = nn.Sequential(OrderedDict(model))
        if kwargs["loss"] == "gaussian":
            covariance = CovarianceMatrix(
                statistics=[kwargs["statistic"]],
                slice_filters=kwargs.get("slice_filters",None),
                select_filters=kwargs.get("select_filters",None),
                normalize_covariance=kwargs["normalize_covariance"]
                if kwargs.get("normalization_dict", None) is not None
                else False,
                normalization_dict=kwargs.get("normalization_dict", None),
            ).get_covariance_data(
                volume_scaling=8.0,
            )
            self.register_buffer(
                "covariance",
                Tensor(
                    covariance.astype(np.float32),
                ),
            )
            self.loss = GaussianNLoglike(
                covariance=self.covariance,
            )
        elif kwargs["loss"] == "learned_gaussian":
            self.loss = nn.GaussianNLLLoss()
        elif kwargs["loss"] == "mse":
            self.loss = nn.MSELoss()
        elif kwargs["loss"] == "mae":
            self.loss = nn.L1Loss()
        for key in ['normalization_dict', 'slice_filters', 'select_filters']:
            if key in kwargs:
                kwargs.pop(key)
        self.save_hyperparameters()

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
        vargs = vars(args)
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

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward model

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output
        """
        return self.mlp(x)

    def _compute_loss(self, batch, batch_idx) -> float:
        """Compute loss in batch

        Args:
            batch: batch with x and y
            batch_idx: batch idx

        Returns:
            float: loss
        """
        x, y = batch
        y_pred = self.forward(x)
        return self.loss(y, y_pred)
