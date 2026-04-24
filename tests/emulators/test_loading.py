import pickle
from pathlib import Path

import pytest
import torch

from sunbird.emulators import loading


class LoadedModel:
    def __init__(self):
        self.eval_called = False
        self.device = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


def make_fake_model_class(calls):
    class FakeModel:
        __name__ = "FakeModel"

        @classmethod
        def load_from_checkpoint(cls, checkpoint_fn, strict=True, **kwargs):
            calls.append(
                {
                    "checkpoint_fn": Path(checkpoint_fn),
                    "strict": strict,
                    "kwargs": kwargs,
                }
            )
            return LoadedModel()

    return FakeModel


def test_load_model_from_checkpoint_uses_explicit_model_class(monkeypatch, tmp_path):
    checkpoint_fn = tmp_path / "model.ckpt"
    checkpoint_fn.write_bytes(b"not a torch checkpoint")
    calls = []
    fake_model_cls = make_fake_model_class(calls)

    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("Explicit model_cls should skip checkpoint inspection.")

    monkeypatch.setattr(loading, "_get_model_class_from_checkpoint", unexpected_inspection)

    model = loading.load_model_from_checkpoint(
        checkpoint_fn,
        model_cls=fake_model_cls,
        strict=False,
    )

    assert model.eval_called is True
    assert model.device == "cpu"
    assert calls == [
        {
            "checkpoint_fn": checkpoint_fn,
            "strict": False,
            "kwargs": {},
        }
    ]


@pytest.mark.parametrize(
    ("model_type", "model_key"),
    [
        ("fcn", "fcn"),
        ("transformer", "transformer"),
        ("zhong24_transformer", "zhong24_transformer"),
    ],
)
def test_load_model_from_checkpoint_dispatches_from_model_type(
    monkeypatch,
    tmp_path,
    model_type,
    model_key,
):
    checkpoint_fn = tmp_path / "model.ckpt"
    torch.save({"hyper_parameters": {"model_type": model_type}}, checkpoint_fn)
    calls = []
    fake_model_cls = make_fake_model_class(calls)
    monkeypatch.setitem(loading.MODEL_TYPES, model_key, fake_model_cls)

    model = loading.load_model_from_checkpoint(checkpoint_fn)

    assert model.eval_called is True
    assert model.device == "cpu"
    assert calls == [
        {
            "checkpoint_fn": checkpoint_fn,
            "strict": True,
            "kwargs": {},
        }
    ]


def test_load_model_from_checkpoint_defaults_missing_model_type_to_fcn(
    monkeypatch,
    tmp_path,
):
    checkpoint_fn = tmp_path / "model.ckpt"
    torch.save({"hyper_parameters": {}}, checkpoint_fn)
    calls = []
    fake_model_cls = make_fake_model_class(calls)
    monkeypatch.setitem(loading.MODEL_TYPES, "fcn", fake_model_cls)

    loading.load_model_from_checkpoint(checkpoint_fn)

    assert calls[0]["checkpoint_fn"] == checkpoint_fn


def test_load_model_from_checkpoint_rejects_unknown_model_type(tmp_path):
    checkpoint_fn = tmp_path / "model.ckpt"
    torch.save({"hyper_parameters": {"model_type": "mystery"}}, checkpoint_fn)

    with pytest.raises(ValueError, match="Unknown emulator model_type 'mystery'"):
        loading.load_model_from_checkpoint(checkpoint_fn)


def test_load_model_from_checkpoint_registers_safe_globals(monkeypatch, tmp_path):
    checkpoint_fn = tmp_path / "model.ckpt"
    checkpoint_fn.write_bytes(b"not a torch checkpoint")
    calls = []
    fake_model_cls = make_fake_model_class(calls)
    registered = {}

    def fake_add_safe_globals(safe_globals):
        registered["safe_globals"] = safe_globals

    monkeypatch.setattr(torch.serialization, "add_safe_globals", fake_add_safe_globals)

    loading.load_model_from_checkpoint(checkpoint_fn, model_cls=fake_model_cls)

    assert registered["safe_globals"] == loading.SAFE_GLOBALS


def test_load_model_from_checkpoint_retries_weights_only_load(
    monkeypatch,
    tmp_path,
):
    checkpoint_fn = tmp_path / "model.ckpt"
    checkpoint_fn.write_bytes(b"not a torch checkpoint")
    calls = []

    class FakeModel:
        __name__ = "FakeModel"

        @classmethod
        def load_from_checkpoint(cls, checkpoint_fn, strict=True, **kwargs):
            calls.append(
                {
                    "checkpoint_fn": Path(checkpoint_fn),
                    "strict": strict,
                    "kwargs": kwargs,
                }
            )
            if len(calls) == 1:
                raise pickle.UnpicklingError("Weights only load failed")
            return LoadedModel()

    model = loading.load_model_from_checkpoint(checkpoint_fn, model_cls=FakeModel)

    assert model.eval_called is True
    assert calls == [
        {
            "checkpoint_fn": checkpoint_fn,
            "strict": True,
            "kwargs": {},
        },
        {
            "checkpoint_fn": checkpoint_fn,
            "strict": True,
            "kwargs": {"weights_only": False},
        },
    ]


def test_load_model_from_checkpoint_retries_weights_only_inspection(
    monkeypatch,
    tmp_path,
):
    checkpoint_fn = tmp_path / "model.ckpt"
    calls = []
    model_calls = []
    fake_model_cls = make_fake_model_class(model_calls)
    monkeypatch.setitem(loading.MODEL_TYPES, "fcn", fake_model_cls)

    def fake_torch_load(path, **kwargs):
        calls.append({"path": Path(path), "kwargs": kwargs})
        if len(calls) == 1:
            raise pickle.UnpicklingError("Weights only load failed")
        return {"hyper_parameters": {"model_type": "fcn"}}

    monkeypatch.setattr(loading.torch, "load", fake_torch_load)

    loading.load_model_from_checkpoint(checkpoint_fn)

    assert calls == [
        {
            "path": checkpoint_fn,
            "kwargs": {"map_location": torch.device("cpu")},
        },
        {
            "path": checkpoint_fn,
            "kwargs": {
                "map_location": torch.device("cpu"),
                "weights_only": False,
            },
        },
    ]
    assert model_calls[0]["checkpoint_fn"] == checkpoint_fn
