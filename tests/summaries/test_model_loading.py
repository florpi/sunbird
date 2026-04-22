from pathlib import Path
from types import SimpleNamespace

import pytest

from sunbird.summaries.base import BaseSummary


def test_load_model_dispatches_transformer_from_hparams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "hparams.yaml").write_text("model_type: transformer\n")
    sentinel_model = SimpleNamespace(eval=lambda: sentinel_model)

    def unexpected(*args, **kwargs):
        raise AssertionError("FCN path should not be used for transformer models.")

    called = {}

    def fake_from_folder(path_to_model, load_loss=False):
        called["path_to_model"] = path_to_model
        called["load_loss"] = load_loss
        return sentinel_model

    monkeypatch.setattr(
        "sunbird.summaries.base.FCN.from_folder",
        unexpected,
    )
    monkeypatch.setattr(
        "sunbird.summaries.base.Transformer.from_folder",
        fake_from_folder,
    )

    model, flax_params = BaseSummary.load_model(tmp_path, flax=False)

    assert model is sentinel_model
    assert flax_params is None
    assert called == {
        "path_to_model": tmp_path,
        "load_loss": False,
    }


def test_load_model_rejects_flax_for_transformer(
    tmp_path: Path,
):
    (tmp_path / "hparams.yaml").write_text("model_type: transformer\n")

    with pytest.raises(NotImplementedError, match="Flax/JAX"):
        BaseSummary.load_model(tmp_path, flax=True)


def test_load_model_defaults_to_fcn_when_model_type_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "hparams.yaml").write_text("n_hidden: [64, 64]\n")
    sentinel_model = SimpleNamespace(eval=lambda: sentinel_model)
    called = {}

    def fake_from_folder(path_to_model, load_loss=False):
        called["path_to_model"] = path_to_model
        called["load_loss"] = load_loss
        return sentinel_model

    monkeypatch.setattr(
        "sunbird.summaries.base.FCN.from_folder",
        fake_from_folder,
    )
    monkeypatch.setattr(
        "sunbird.summaries.base.Transformer.from_folder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Transformer path should not be used without model_type.")
        ),
    )

    model, flax_params = BaseSummary.load_model(tmp_path, flax=False)

    assert model is sentinel_model
    assert flax_params is None
    assert called == {
        "path_to_model": tmp_path,
        "load_loss": False,
    }


def test_load_model_dispatches_zhong24_transformer_from_hparams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "hparams.yaml").write_text("model_type: zhong24_transformer\n")
    sentinel_model = SimpleNamespace(eval=lambda: sentinel_model)

    called = {}

    def fake_from_folder(path_to_model, load_loss=False):
        called["path_to_model"] = path_to_model
        called["load_loss"] = load_loss
        return sentinel_model

    monkeypatch.setattr(
        "sunbird.summaries.base.FCN.from_folder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FCN path should not be used for zhong24 transformer models.")
        ),
    )
    monkeypatch.setattr(
        "sunbird.summaries.base.Transformer.from_folder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Generic transformer path should not be used for zhong24 transformer models.")
        ),
    )
    monkeypatch.setattr(
        "sunbird.summaries.base.Zhong24Transformer.from_folder",
        fake_from_folder,
    )

    model, flax_params = BaseSummary.load_model(tmp_path, flax=False)

    assert model is sentinel_model
    assert flax_params is None
    assert called == {
        "path_to_model": tmp_path,
        "load_loss": False,
    }


def test_load_model_rejects_flax_for_zhong24_transformer(
    tmp_path: Path,
):
    (tmp_path / "hparams.yaml").write_text("model_type: zhong24_transformer\n")

    with pytest.raises(NotImplementedError, match="Flax/JAX"):
        BaseSummary.load_model(tmp_path, flax=True)
