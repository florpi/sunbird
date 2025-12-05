import torch
import numpy as np
from torch import nn, Tensor
from sunbird.emulators.models import BaseModel
from sunbird.emulators.models.activation import LearnedSigmoid
from sunbird.emulators.loss import MultivariateGaussianNLLLoss, GaussianNLoglike, WeightedL1Loss, WeightedMSELoss

class Transformer(BaseModel):
    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate: float = 0.,
        learning_rate: float = 1.e-3,
        scheduler_patience: int = 30,
        scheduler_factor: int = 0.5,
        scheduler_threshold: int = 1.e-6,
        weight_decay: float = 0.,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        loss: str = 'rmse',
        training: bool = True,
        *args, 
        **kwargs,
    ):
        super(Transformer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_threshold = scheduler_threshold
        self.weight_decay = weight_decay
        self.loss = loss
        
        # Embedding layer (for the input sequence of cosmological parameters)
        self.embedding = nn.Linear(n_input, d_model)
        
        # Transformer Encoder Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward
            ), num_layers=num_layers
        )
        
        # Final linear layer to map the transformer output to the desired output
        self.fc_out = nn.Linear(d_model, n_output)

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

    @property
    def flax_attributes(self,):
        return {'n_input': self.n_input,
                'n_hidden': self.n_hidden,
                'act_fn': self.act_fn_str,
                'n_output': self.n_output,
                'predict_errors': True if self.loss == "learned_gaussian" else False,
                'transform_output': self.transform_output,
                'coordinates': self.coordinates,
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

    def forward(self, x):
        if x.dim() == 2:  # If x has shape (batch_size, d_model)
            x = x.unsqueeze(1)
        x = self.embedding(x)  # shape (batch_size, 1, d_model)
        x = x.permute(1, 0, 2)  # shape (1, batch_size, d_model) for transformer input
        
        # Pass through the transformer
        x = self.transformer(x)
        
        # Take the output of the first (and only) token
        x = x[0, :, :]
        
        # Output layer
        x = self.fc_out(x)
        x_var = torch.zeros_like(x)
        return x, x_var

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