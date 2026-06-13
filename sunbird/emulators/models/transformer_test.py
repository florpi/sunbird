import torch
import numpy as np
from torch import nn, Tensor
from typing import OrderedDict, Dict, List, Optional
from sunbird.emulators.models import BaseModel
from sunbird.emulators.models.activation import LearnedSigmoid
from sunbird.emulators.loss import MultivariateGaussianNLLLoss, GaussianNLoglike, get_cholesky_decomposition_covariance, WeightedL1Loss, WeightedMSELoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(BaseModel):
    def __init__(
            self, 
            n_input,
            n_output,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=128,
            dropout_rate=0.1,
            learning_rate=1.e-3,
            scheduler_patience=30,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=0.,
            act_fn='learned_sigmoid',
            loss='rmse',
            training=True,
            mean_input=None,
            std_input=None,
            mean_output=None,
            std_output=None,
            standarize_input=True,
            standarize_output=True,
            transform_input=None,
            transform_output=None,
            coordinates=None,
            compression_matrix=None,
            *args, 
            **kwargs,
    ):
        """Transformer-based neural network for power spectrum emulation"""
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_threshold = scheduler_threshold
        self.weight_decay = weight_decay
        self.act_fn_str = act_fn
        self.coordinates = coordinates
        self.standarize_input = standarize_input
        self.standarize_output = standarize_output
        
        if n_output <= 0:
            raise ValueError(f"n_output must be > 0, got {n_output}")
        
        self.register_parameter('mean_input', mean_input, n_input)
        self.register_parameter('std_input', std_input, n_input)
        self.register_parameter('mean_output', mean_output, n_output)
        self.register_parameter('std_output', std_output, n_output)
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.loss = loss
        self.data_dim = self.n_output
        if self.loss == "learned_gaussian":
            self.n_output *= 2
        elif self.loss == "multivariate_learned_gaussian":
            self.n_output += (self.n_output*(self.n_output+1))//2

        # Fixed number of tokens for the transformer sequence
        self.num_tokens = 32
        self.token_embeddings = nn.Embedding(self.num_tokens, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_tokens)
        
        # Project input parameters to conditioning vector
        self.input_projection = nn.Linear(n_input, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection from final token representation
        self.output_projection = nn.Linear(d_model, self.n_output)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Model arguments that could vary

        Args:
            parent_parser (parser): parser

        Returns:
            parser: updated parser
        """
        parser = parent_parser.add_argument_group("Transformer")
        parser.add_argument("--d_model", type=int, default=32)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--dim_feedforward", type=int, default=128)
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
    ) -> "Transformer":
        """Create transformer from parsed args

        Args:
            args (args): command line arguments

        Returns:
            Transformer: transformer model
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
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'n_output': self.n_output,
                'predict_errors': True if self.loss == "learned_gaussian" else False,
                'transform_output': self.transform_output,
                'coordinates': self.coordinates,
                'compression_matrix': None,
        }

    def register_parameter(self, parameter_name, parameter, dim):
        if parameter is not None:
            self.register_buffer(parameter_name, torch.tensor(parameter, dtype=torch.float32))
        else:
            self.register_buffer(parameter_name, torch.ones((dim,), dtype=torch.float32))

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
            x (Tensor): input tensor (batch_size, n_input)

        Returns:
            Tensor: output (predictions, variance)
        """
        if self.standarize_input:
            std_input = self.std_input.to(x.device)
            mean_input = self.mean_input.to(x.device)
            x = (x - mean_input) / std_input
        
        batch_size = x.size(0)
        
        # Get learned token embeddings (fixed sequence length)
        tokens = self.token_embeddings(torch.arange(self.num_tokens, device=x.device))  # (num_tokens, d_model)
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_tokens, d_model)
        
        # Project input to conditioning vector
        conditioning = self.input_projection(x)  # (batch_size, d_model)
        conditioning = conditioning.unsqueeze(1).expand(-1, self.num_tokens, -1)  # (batch_size, num_tokens, d_model)
        
        # Condition tokens via broadcast addition
        conditioned_tokens = tokens + conditioning  # (batch_size, num_tokens, d_model)
        
        # Add positional encoding
        conditioned_tokens = conditioned_tokens.permute(1, 0, 2)  # (num_tokens, batch_size, d_model)
        conditioned_tokens = self.pos_encoder(conditioned_tokens)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(conditioned_tokens)  # (num_tokens, batch_size, d_model)
        
        # Use the final token's representation for output
        final_repr = transformer_out[-1, :, :]  # (batch_size, d_model)
        
        # Project to output
        output = self.output_projection(final_repr)  # (batch_size, self.n_output)
        
        if self.loss == "learned_gaussian":
            y_pred, y_var = torch.chunk(output, 2, dim=-1)
            y_var = nn.Softplus()(y_var)
        elif self.loss == "multivariate_learned_gaussian":
            y_cov = output[..., self.data_dim:]
            y_pred = output[..., :self.data_dim]
            L = get_cholesky_decomposition_covariance(y_cov, data_dim=self.data_dim,)
            y_var = L
        else:
            y_pred = output
            y_var = torch.zeros_like(y_pred)
                
        return y_pred, y_var

    def get_prediction(self, x: Tensor, filters: Optional[dict] = None) -> Tensor:
        x = torch.Tensor(x)
        if self.transform_input:
            x = self.transform_input.transform(x)
        y, _ = self.forward(x) 
        if self.standarize_output:
            std_output = self.std_output.to(x.device)
            mean_output = self.mean_output.to(x.device)
            y =  y * std_output + mean_output
        if self.transform_output:
            y = self.transform_output.inverse_transform(y)
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