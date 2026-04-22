import numpy as np
import pytest
import torch
from types import MethodType

from sunbird.emulators import Transformer


def test_transformer_forward_returns_expected_shapes():
    model = Transformer(
        n_input=5,
        n_output=3,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dim_feedforward=32,
        dropout_rate=0.0,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        loss="weighted_mae",
        training=True,
        mean_input=np.zeros(5),
        std_input=np.ones(5),
        mean_output=np.zeros(3),
        std_output=np.ones(3),
        covariance_matrix=np.eye(3),
    )

    y_pred, y_var = model.forward(torch.randn(7, 5))

    assert y_pred.shape == (7, 3)
    assert y_var.shape == (7, 3)


def test_transformer_get_prediction_applies_output_rescaling():
    model = Transformer(
        n_input=4,
        n_output=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout_rate=0.0,
        loss="mae",
        training=False,
        mean_input=np.zeros(4),
        std_input=np.ones(4),
        mean_output=np.array([10.0, 20.0]),
        std_output=np.array([2.0, 3.0]),
    )

    def fake_forward(self, x):
        y = torch.ones((x.shape[0], 2), dtype=torch.float32, device=x.device)
        return y, torch.zeros_like(y)

    model.forward = MethodType(fake_forward, model)

    prediction = model.get_prediction(torch.zeros((3, 4), dtype=torch.float32))

    expected = torch.tensor([[12.0, 23.0]] * 3, dtype=torch.float32)
    assert torch.allclose(prediction, expected)


def test_transformer_get_prediction_accepts_single_feature_vector():
    model = Transformer(
        n_input=4,
        n_output=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout_rate=0.0,
        loss="mae",
        training=False,
        mean_input=np.zeros(4),
        std_input=np.ones(4),
        mean_output=np.array([10.0, 20.0]),
        std_output=np.array([2.0, 3.0]),
    )

    def fake_forward(self, x):
        y = torch.ones((x.shape[0], 2), dtype=torch.float32, device=x.device)
        return y, torch.zeros_like(y)

    model.forward = MethodType(fake_forward, model)

    prediction = model.get_prediction(torch.zeros(4, dtype=torch.float32))

    expected = torch.tensor([12.0, 23.0], dtype=torch.float32)
    assert torch.allclose(prediction, expected)


def test_transformer_compute_loss_uses_configured_loss():
    model = Transformer(
        n_input=3,
        n_output=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout_rate=0.0,
        loss="mae",
        training=True,
        standarize_output=False,
        covariance_matrix=np.eye(2),
    )

    def fake_forward(self, x):
        y = torch.ones((x.shape[0], 2), dtype=torch.float32, device=x.device)
        return y, torch.zeros_like(y)

    model.forward = MethodType(fake_forward, model)
    batch = (
        torch.zeros((4, 3), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
    )

    loss = model._compute_loss(batch=batch, batch_idx=0)

    assert float(loss) == pytest.approx(1.0)


def test_transformer_saves_model_type_in_hparams():
    model = Transformer(
        n_input=3,
        n_output=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout_rate=0.0,
        loss="mae",
        training=True,
        covariance_matrix=np.eye(2),
    )

    assert model.hparams["model_type"] == "transformer"


def test_transformer_requires_divisible_model_width():
    with pytest.raises(ValueError, match="must be divisible"):
        Transformer(
            n_input=3,
            n_output=2,
            d_model=10,
            n_heads=4,
            n_layers=1,
            dim_feedforward=16,
            dropout_rate=0.0,
            loss="mae",
            training=False,
        )
