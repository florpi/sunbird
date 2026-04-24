import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch

from sunbird.data.transforms_array import ArcsinhTransform, LogTransform
from sunbird.emulators.models import BaseModel, FCN, Transformer, Zhong24Transformer

logger = logging.getLogger(__name__)

SAFE_GLOBALS = [LogTransform, ArcsinhTransform]
MODEL_TYPES = {
    "fcn": FCN,
    "transformer": Transformer,
    "zhong24_transformer": Zhong24Transformer,
}


def _is_weights_only_error(error: pickle.UnpicklingError) -> bool:
    return "Weights only load failed" in str(error)


def _register_safe_globals() -> None:
    try:
        torch.serialization.add_safe_globals(SAFE_GLOBALS)
    except AttributeError:
        logger.debug(
            "torch.serialization.add_safe_globals is not available; "
            "skipping safe globals registration."
        )


def _load_checkpoint_payload(checkpoint_fn: Union[Path, str]) -> Dict[str, Any]:
    checkpoint_fn = Path(checkpoint_fn)
    try:
        return torch.load(checkpoint_fn, map_location=torch.device("cpu"))
    except pickle.UnpicklingError as error:
        if not _is_weights_only_error(error):
            raise
        logger.warning(
            "Retrying checkpoint inspection with weights_only=False for %s "
            "due to PyTorch weights-only unpickling restrictions.",
            checkpoint_fn,
        )
        return torch.load(
            checkpoint_fn,
            map_location=torch.device("cpu"),
            weights_only=False,
        )


def _get_model_class_from_checkpoint(
    checkpoint_fn: Union[Path, str],
) -> Type[BaseModel]:
    checkpoint = _load_checkpoint_payload(checkpoint_fn)
    hparams = checkpoint.get("hyper_parameters", {})
    model_type = str(hparams.get("model_type", "fcn")).lower()
    try:
        return MODEL_TYPES[model_type]
    except KeyError as error:
        known_types = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(
            f"Unknown emulator model_type '{model_type}' in checkpoint {checkpoint_fn}. "
            f"Known types are: {known_types}."
        ) from error


def load_model_from_checkpoint(
    checkpoint_fn: Union[Path, str],
    model_cls: Optional[Type[BaseModel]] = None,
    strict: bool = True,
) -> BaseModel:
    """
    Load a Sunbird emulator model from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint_fn
        Path to the model checkpoint file.
    model_cls
        Explicit model class to load. If not provided, the class is inferred from
        ``hyper_parameters.model_type`` in the checkpoint, defaulting to FCN for
        older checkpoints without ``model_type``.
    strict
        Whether to strictly enforce that the checkpoint keys match the model.

    Returns
    -------
    BaseModel
        The loaded model in evaluation mode on CPU.
    """
    checkpoint_fn = Path(checkpoint_fn)
    _register_safe_globals()
    if model_cls is None:
        model_cls = _get_model_class_from_checkpoint(checkpoint_fn)

    logger.info(
        "Loading emulator model from %s with %s",
        checkpoint_fn,
        model_cls.__name__,
    )
    try:
        model = model_cls.load_from_checkpoint(checkpoint_fn, strict=strict)
    except pickle.UnpicklingError as error:
        if not _is_weights_only_error(error):
            raise
        logger.warning(
            "Retrying checkpoint load with weights_only=False for %s "
            "due to PyTorch weights-only unpickling restrictions.",
            checkpoint_fn,
        )
        model = model_cls.load_from_checkpoint(
            checkpoint_fn,
            strict=strict,
            weights_only=False,
        )
    model.eval().to("cpu")
    return model
