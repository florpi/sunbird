import numpy as np
import torch
from torch import Tensor, nn
from typing import Optional, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sunbird.emulators.loss import (
    GaussianNLoglike,
    MultivariateGaussianNLLLoss,
    WeightedL1Loss,
    WeightedMSELoss,
    get_cholesky_decomposition_covariance,
)
from sunbird.emulators.models import BaseModel


class Zhong24TransformerBlock(nn.Module):
    """Post-norm residual transformer block used by Zhong et al. (2024)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_norm = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
        )
        self.feedforward_dropout = nn.Dropout(dropout_rate)
        self.feedforward_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + self.attention_dropout(attended))
        updates = self.feedforward(tokens)
        return self.feedforward_norm(tokens + self.feedforward_dropout(updates))


class Zhong24Transformer(BaseModel):
    """EMC-tuned Zhong-style transformer architecture."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_tokens: int = 10,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 192,
        dropout_rate: float = 0.05,
        learning_rate: float = 5.0e-4,
        weight_decay: float = 1.0e-2,
        scheduler_patience: int = 8,
        scheduler_factor: float = 0.5,
        scheduler_threshold: float = 1.0e-4,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        loss: str = "rmse",
        training: bool = True,
        mean_input: Optional[torch.Tensor] = None,
        std_input: Optional[torch.Tensor] = None,
        mean_output: Optional[torch.Tensor] = None,
        std_output: Optional[torch.Tensor] = None,
        standarize_input: bool = True,
        standarize_output: bool = True,
        transform_input: Optional[callable] = None,
        transform_output: Optional[callable] = None,
        coordinates: Optional[dict] = None,
        compression_matrix: Optional[torch.Tensor] = None,
        model_type: str = "zhong24_transformer",
        *args,
        **kwargs,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}."
            )

        self.n_input = n_input
        self.n_output = n_output
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_threshold = scheduler_threshold
        self.adam_betas = adam_betas
        self.loss = loss
        self.coordinates = coordinates
        self.standarize_input = standarize_input
        self.standarize_output = standarize_output
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.model_type = model_type
        self.data_dim = self.n_output

        self.register_stat_buffer("mean_input", mean_input, n_input)
        self.register_stat_buffer("std_input", std_input, n_input)
        self.register_stat_buffer("mean_output", mean_output, n_output)
        self.register_stat_buffer("std_output", std_output, n_output)

        if self.loss == "learned_gaussian":
            self.n_output *= 2
        elif self.loss == "multivariate_learned_gaussian":
            self.n_output += (self.n_output * (self.n_output + 1)) // 2

        self.embedding = nn.Linear(n_input, n_tokens * d_model)
        self.embedding_activation = nn.PReLU()
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                Zhong24TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(n_tokens * d_model, self.n_output)
        self.reset_parameters()

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

    @property
    def flax_attributes(self):
        raise NotImplementedError(
            "Zhong24Transformer does not currently support Flax/JAX conversion."
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.adam_betas,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            threshold=self.scheduler_threshold,
            threshold_mode="abs",
            min_lr=1.0e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def register_stat_buffer(self, parameter_name, parameter, dim):
        if parameter is not None:
            self.register_buffer(
                parameter_name,
                torch.tensor(parameter, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                parameter_name,
                torch.ones((dim,), dtype=torch.float32),
            )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def load_loss(self, loss: str, **kwargs):
        if "weighted" in loss:
            covariance = kwargs["covariance_matrix"]
            covariance = Tensor(covariance.astype(np.float32))
            if loss == "weighted_mae":
                self.loss_fct = WeightedL1Loss(
                    variance=torch.sqrt(torch.diagonal(covariance))
                )
            elif loss == "weighted_mse":
                self.loss_fct = WeightedMSELoss(
                    variance=torch.diagonal(covariance)
                )
        elif loss == "GaussianNLoglike":
            covariance = kwargs["covariance_matrix"]
            covariance = Tensor(covariance.astype(np.float32))
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

    def forward(self, x: Tensor):
        squeeze_batch = x.ndim == 1
        if squeeze_batch:
            x = x.unsqueeze(0)

        if self.standarize_input:
            std_input = self.std_input.to(x.device)
            mean_input = self.mean_input.to(x.device)
            x = (x - mean_input) / std_input

        tokens = self.embedding(x).reshape(x.shape[0], self.n_tokens, self.d_model)
        tokens = self.embedding_dropout(self.embedding_activation(tokens))
        for block in self.blocks:
            tokens = block(tokens)

        y_pred = self.head(tokens.reshape(tokens.shape[0], -1))

        if self.loss == "learned_gaussian":
            y_pred, y_var = torch.chunk(y_pred, 2, dim=-1)
            y_var = nn.Softplus()(y_var)
            if squeeze_batch:
                y_pred = y_pred.squeeze(0)
                y_var = y_var.squeeze(0)
            return y_pred, y_var
        if self.loss == "multivariate_learned_gaussian":
            y_cov = y_pred[..., self.data_dim :]
            y_pred = y_pred[..., : self.data_dim]
            cholesky = get_cholesky_decomposition_covariance(
                y_cov,
                data_dim=self.data_dim,
            )
            if squeeze_batch:
                y_pred = y_pred.squeeze(0)
                cholesky = cholesky.squeeze(0)
            return y_pred, cholesky

        y_var = torch.zeros_like(y_pred)
        if squeeze_batch:
            y_pred = y_pred.squeeze(0)
            y_var = y_var.squeeze(0)
        return y_pred, y_var

    def get_prediction(
        self,
        x: Tensor,
        filters: Optional[dict] = None,
        skip_output_inverse_transform: bool = False,
    ) -> Tensor:
        x = torch.Tensor(x)
        if self.transform_input is not None:
            x = self.transform_input.transform(x)
        y, _ = self.forward(x)
        if self.standarize_output:
            std_output = self.std_output.to(x.device)
            mean_output = self.mean_output.to(x.device)
            y = y * std_output + mean_output
        if self.transform_output is not None and not skip_output_inverse_transform:
            y = self.transform_output.inverse_transform(y)
        if self.compression_matrix is not None:
            y = y @ self.compression_matrix
        return y

    def _compute_loss(self, batch, batch_idx) -> float:
        x, y = batch
        y_pred, y_var = self.forward(x)
        if self.standarize_output:
            std_output = self.std_output.to(x.device)
            mean_output = self.mean_output.to(x.device)
            y_pred = y_pred * std_output + mean_output
        if self.loss in {"learned_gaussian", "multivariate_learned_gaussian"}:
            return self.loss_fct(y_pred, y, y_var)
        return self.loss_fct(y, y_pred)
