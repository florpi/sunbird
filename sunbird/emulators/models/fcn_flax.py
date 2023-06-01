from typing import Tuple, Sequence, Dict
from pathlib import Path
import numpy as np
import yaml
import jax
import torch
from flax.traverse_util import unflatten_dict
import jax.numpy as jnp
import flax.linen as nn


def convert_state_dict_from_pt(
    model,
    state,
):
    """
    Converts a PyTorch parameter state dict to an equivalent Flax parameter state dict
    """
    state = {k: v.numpy() for k, v in state.items()}
    state = model.convert_from_pytorch(
        state,
    )
    state = unflatten_dict({tuple(k.split(".")): v for k, v in state.items()})
    return state


class FlaxFCN(nn.Module):
    """Simple fully connected flax version of the emulator"""

    n_input: int
    n_hidden: Sequence[int]
    act_fn: str
    n_output: int
    predict_errors: False

    def setup(
        self,
    ):
        """Setup the activation function from strings"""
        self.actvation_fn = getattr(nn, self.act_fn.lower())

    @classmethod
    def from_folder(cls, path_to_model: Path) -> Tuple["FlaxFCN", Dict]:
        """get the model and weights from a folder

        Args:
            path_to_model (Path): path to the folder

        Returns:
            Tuple: model and its parameters
        """
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        if config["loss"] == "learned_gaussian":
            n_output = config["n_output"] * 2
        else:
            n_output = config["n_output"]
        nn_model = cls(
            n_input=config["n_input"],
            n_hidden=config["n_hidden"],
            act_fn=config["act_fn"],
            n_output=n_output,
            predict_errors=True if config["loss"] == "learned_gaussian" else False,
        )
        files = list((path_to_model / "checkpoints").glob("*.ckpt"))
        file_idx = np.argmin(
            [float(str(file).split(".ckpt")[0].split("=")[-1]) for file in files]
        )
        weights_dict = torch.load(
            files[file_idx],
            map_location=torch.device("cpu"),
        )
        state_dict = weights_dict["state_dict"]
        flax_params = convert_state_dict_from_pt(
            model=nn_model,
            state=state_dict,
        )
        return nn_model, flax_params

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        """forward pass

        Args:
            x (jnp.array): inputs

        Returns:
            jnp.array: outputs
        """
        for i, dims in enumerate(self.n_hidden):
            x = nn.Dense(dims)(x)
            x = self.actvation_fn(x)
        y_pred = nn.Dense(self.n_output)(x)
        if self.predict_errors:
            y_pred, y_var = np.split(y_pred, 2, axis=-1)
            y_var = nn.softplus(y_var)
        else:
            y_var = jnp.zeros_like(y_pred)
        return y_pred, y_var

    def convert_from_pytorch(self, pt_state: Dict) -> Dict:
        """Convert the state dict from pytorch to flax

        Args:
            pt_state (Dict): state dictionary with model weights

        Returns:
            Dict: flax weights
        """
        jax_state = dict(pt_state)
        for key, tensor in pt_state.items():
            if "mlp" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                key = key.replace("mlp.mlp", f"Dense_")
                jax_state[key] = tensor.T
        return jax_state
