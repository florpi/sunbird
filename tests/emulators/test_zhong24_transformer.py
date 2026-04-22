import numpy as np
import pytest
import torch
from types import MethodType
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sunbird.emulators import Zhong24Transformer


def test_zhong24_transformer_forward_returns_expected_shapes():
    model = Zhong24Transformer(
        n_input=5,
        n_output=3,
        n_tokens=25,
        d_model=20,
        n_heads=4,
        n_layers=2,
        dim_feedforward=20,
        dropout_rate=0.0,
        learning_rate=5.0e-4,
        weight_decay=1.0e-2,
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


def test_zhong24_transformer_get_prediction_applies_output_rescaling():
    model = Zhong24Transformer(
        n_input=4,
        n_output=2,
        d_model=20,
        n_heads=4,
        n_layers=1,
        dim_feedforward=20,
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


def test_zhong24_transformer_get_prediction_accepts_single_feature_vector():
    model = Zhong24Transformer(
        n_input=4,
        n_output=2,
        d_model=20,
        n_heads=4,
        n_layers=1,
        dim_feedforward=20,
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


def test_zhong24_transformer_saves_model_type_in_hparams():
    model = Zhong24Transformer(
        n_input=3,
        n_output=2,
        d_model=20,
        n_heads=4,
        n_layers=1,
        dim_feedforward=20,
        dropout_rate=0.0,
        loss="mae",
        training=True,
        covariance_matrix=np.eye(2),
    )

    assert model.hparams["model_type"] == "zhong24_transformer"


def test_zhong24_transformer_uses_small_data_defaults():
    model = Zhong24Transformer(
        n_input=3,
        n_output=2,
        loss="mae",
        training=False,
    )

    assert model.n_tokens == 10
    assert model.d_model == 96
    assert model.n_layers == 2
    assert model.dim_feedforward == 192
    assert model.dropout_rate == pytest.approx(0.05)
    assert model.scheduler_patience == 8
    assert model.scheduler_factor == pytest.approx(0.5)
    assert model.scheduler_threshold == pytest.approx(1.0e-4)


def test_zhong24_transformer_configure_optimizers_sets_scheduler():
    model = Zhong24Transformer(
        n_input=3,
        n_output=2,
        loss="mae",
        training=False,
        scheduler_patience=9,
        scheduler_factor=0.25,
        scheduler_threshold=2.0e-4,
    )

    config = model.configure_optimizers()

    assert isinstance(config["optimizer"], AdamW)
    assert config["lr_scheduler"]["monitor"] == "val_loss"
    assert isinstance(config["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert config["lr_scheduler"]["scheduler"].patience == 9
    assert config["lr_scheduler"]["scheduler"].factor == pytest.approx(0.25)
    assert config["lr_scheduler"]["scheduler"].threshold == pytest.approx(2.0e-4)


def test_zhong24_transformer_requires_divisible_model_width():
    with pytest.raises(ValueError, match="must be divisible"):
        Zhong24Transformer(
            n_input=3,
            n_output=2,
            d_model=22,
            n_heads=4,
            n_layers=1,
            dim_feedforward=20,
            dropout_rate=0.0,
            loss="mae",
            training=False,
        )
