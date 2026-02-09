from typing import OrderedDict, Dict, List, Optional
import numpy as np
from torch import nn, Tensor
import torch

from sunbird.emulators.models import BaseModel
from sunbird.covariance import CovarianceMatrix
from sunbird.emulators.loss import MultivariateGaussianNLLLoss, GaussianNLoglike, get_cholesky_decomposition_covariance, WeightedL1Loss, WeightedMSELoss
from sunbird.emulators.models.activation import LearnedSigmoid
from sunbird.data.data_utils import convert_to_summary


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class FCN(BaseModel):
    def __init__(
            self, 
            n_input: int,
            n_output: int,
            n_hidden: List[int] =[512, 512, 512, 512],
            dropout_rate: float = 0.,
            learning_rate: float = 1.e-3,
            scheduler_patience: int = 30,
            scheduler_factor: int = 0.5,
            scheduler_threshold: int = 1.e-6,
            weight_decay: float = 0.,
            act_fn: str = 'learned_sigmoid',
            loss: str = 'rmse',
            training: bool = True,
            mean_input: Optional[torch.Tensor] = None,
            std_input: Optional[torch.Tensor] = None,
            mean_output: Optional[torch.Tensor] = None,
            std_output: Optional[torch.Tensor] = None,
            standarize_input: bool = True,
            standarize_output: bool = True,
            input_transform: Optional[callable] = None,
            output_transform: Optional[callable] = None,
            coordinates: Optional[dict] = None,
            compression_matrix: Optional[torch.Tensor] = None,
            *args, 
            **kwargs,
    ):
        """Fully connected neural network with variable architecture"""
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_threshold = scheduler_threshold
        self.weight_decay = weight_decay
        self.act_fn_str = act_fn
        self.coordinates = coordinates
        self.standarize_input = standarize_input
        self.standarize_output= standarize_output
        self.register_parameter('mean_input', mean_input, n_input)
        self.register_parameter('std_input', std_input, n_input)
        self.register_parameter('mean_output', mean_output, n_output)
        self.register_parameter('std_output', std_output, n_output)
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.loss = loss
        self.data_dim = self.n_output
        if self.loss == "learned_gaussian":
            self.n_output *= 2
        elif self.loss == "multivariate_learned_gaussian":
            self.n_output += (self.n_output*(self.n_output+1))//2
        self.mlp = self.get_model(
            n_input=self.n_input,
            n_hidden=self.n_hidden,
            n_output=self.n_output,
            dropout_rate=dropout_rate,
        )
        if training:
            self.load_loss(loss=loss, **kwargs)
            self.save_hyperparameters(
                ignore=[
                    "mean_input",
                    "std_input",
                    "mean_output",
                    "std_output",
                ],
            )
        self.compression_matrix = compression_matrix
    
    def __setattr__(self, name, value):
        """Override to provide backward compatibility for renamed attributes"""
        # Map old attribute names to new ones for backward compatibility
        if name == 'transform_input':
            name = 'input_transform'
        elif name == 'transform_output':
            name = 'output_transform'
        super().__setattr__(name, value)
    
    def __getattr__(self, name):
        """Override to provide backward compatibility for renamed attributes"""
        # Map old attribute names to new ones for backward compatibility
        if name == 'transform_input':
            return self.input_transform
        elif name == 'transform_output':
            return self.output_transform
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
                'output_transform': self.output_transform,
                'coordinates': self.coordinates,
                'compression_matrix': None,
        }

    def register_parameter(self, parameter_name, parameter, dim):
        if parameter is not None:
            self.register_buffer(parameter_name, torch.tensor(parameter, dtype=torch.float32))
        else:
            self.register_buffer(parameter_name, torch.ones((dim,), dtype=torch.float32))

    def get_activation_function(self, layer_index: int) -> nn.Module:
        """ Returns the activation function for the given layer index. """
        if self.act_fn_str == 'learned_sigmoid':
            return LearnedSigmoid(self.n_hidden[layer_index])
        return getattr(nn, self.act_fn_str)()

    def add_layer(self, model, input_dim: int, output_dim: int, layer_index: int, dropout_rate,) -> None:
        """ Creates a single layer for the network. """
        layer_name = f"mlp{layer_index}"
        act_name = f"act{layer_index}"
        dropout_name = f"dropout{layer_index}"

        linear_layer = nn.Linear(input_dim, output_dim)
        activation = self.get_activation_function(layer_index)

        model.add_module(layer_name, linear_layer)
        model.add_module(act_name, activation)
        model.add_module(dropout_name, nn.Dropout(dropout_rate))
        return model

    def get_model(self, n_input: int, n_hidden: List[int], n_output: int, dropout_rate: float) -> nn.Sequential:
        """
        Constructs a multi-layer perceptron model.

        Args:
            n_input (int): Number of input features.
            n_hidden (List[int]): Number of neurons in each hidden layer.
            n_output (int): Number of output features.

        Returns:
            nn.Sequential: The constructed MLP model.
        """
        model = nn.Sequential(OrderedDict())
        last_dim = n_input

        for i, hidden_dim in enumerate(n_hidden):
            model = self.add_layer(model=model, input_dim=last_dim, output_dim=hidden_dim, layer_index=i, dropout_rate=dropout_rate)
            last_dim = hidden_dim

        # Add the final layer
        final_layer_name = f"mlp{len(n_hidden)}"
        model.add_module(final_layer_name, nn.Linear(last_dim, n_output))
        return model

    def load_loss(self, loss: str, **kwargs):
        """Load loss function

        Args:
            loss (str): loss to load
        """
        if "weighted" in loss:
            covariance = kwargs["covariance_matrix"]
            covariance = Tensor(
                covariance.astype(np.float32),
            )
            if loss == "weighted_mae":
                self.loss_fct = WeightedL1Loss(
                    variance=torch.sqrt(torch.diagonal(covariance))
                )
            elif loss == "weighted_mse":
                self.loss_fct = WeightedMSELoss(variance=torch.diagonal(covariance))
        elif loss == 'GaussianNLoglike':
            covariance = kwargs["covariance_matrix"]
            covariance = Tensor(
                covariance.astype(np.float32),
            )
            self.loss_fct = GaussianNLoglike(covariance=covariance)
        elif loss == "learned_gaussian":
            self.loss_fct = nn.GaussianNLLLoss()
        elif loss == "multivariate_learned_gaussian":
            self.loss_fct = MultivariateGaussianNLLLoss()
        elif loss == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss == "rmse":
            self.loss_fct = lambda y, y_pred: torch.sqrt(nn.MSELoss()(y, y_pred))

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
        if self.standarize_input:
            std_input = self.std_input.to(x.device)
            mean_input = self.mean_input.to(x.device)
            x = (x - mean_input) / std_input
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

    def get_prediction(self, x: Tensor, filters: Optional[dict] = None, skip_output_inverse_transform: bool = False) -> Tensor:
        """Get prediction from the model.
        
        Args:
            x (Tensor): Input tensor
            filters (dict, optional): Filters to apply. Defaults to None.
            skip_output_inverse_transform (bool, optional): If True, skip the output inverse transformation,
                keeping predictions in the transformed space. Useful when performing inference in transformed 
                space (requires transforming observations and covariance to match). Defaults to False.
        
        Returns:
            Tensor: Model prediction
        """
        x = torch.Tensor(x)
        if self.input_transform is not None:
            x = self.input_transform.transform(x)
        y, _ = self.forward(x) 
        if self.standarize_output:
            std_output = self.std_output.to(x.device)
            mean_output = self.mean_output.to(x.device)
            y =  y * std_output + mean_output
        if self.output_transform is not None and not skip_output_inverse_transform:
            y = self.output_transform.inverse_transform(y)
        if self.compression_matrix is not None:
            y = y @ self.compression_matrix
        return y

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
        if self.standarize_output:
            std_output = self.std_output.to(x.device)
            mean_output = self.mean_output.to(x.device)
            y_pred = y_pred * std_output + mean_output
        if self.loss == "learned_gaussian":
            return self.loss_fct(y_pred, y, y_var)
        elif self.loss == "multivariate_learned_gaussian":
            return self.loss_fct(y_pred, y, y_var)
        return self.loss_fct(y, y_pred)

