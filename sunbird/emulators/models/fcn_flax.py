from typing import Tuple, Sequence, Dict
from pathlib import Path
import numpy as np
import yaml
import torch
import jax.numpy as jnp
import flax.linen as nn
from sunbird.emulators.models.base import convert_state_dict_from_pt
from sunbird.emulators.models.activation import FlaxLearnedSigmoid
from sunbird.data.data_utils import convert_to_summary



class FlaxFCN(nn.Module):
    """Simple fully connected flax version of the emulator"""

    n_input: int
    n_hidden: Sequence[int]
    act_fn: str
    n_output: int
    predict_errors: False
    transform_output: None
    coordinates: None


    def setup(
        self,
    ):
        pass

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
    def __call__(self, x: jnp.array, filters=None) -> jnp.array:
        """forward pass

        Args:
            x (jnp.array): inputs

        Returns:
            jnp.array: outputs
        """
        mean_input = self.param('mean_input', nn.initializers.zeros, (self.n_input,))
        std_input = self.param('std_input', nn.initializers.ones, (self.n_input,))
        mean_output = self.param('mean_output', nn.initializers.zeros, (self.n_output,))
        std_output = self.param('std_output', nn.initializers.ones, (self.n_output,))
        x = (x - mean_input) / std_input
        for i, dims in enumerate(self.n_hidden):
            x = nn.Dense(dims)(x)
            if self.act_fn == 'learned_sigmoid':
                activation_fn = FlaxLearnedSigmoid(n_dim=x.shape[-1])
            else:
                activation_fn = getattr(nn, self.act_fn.lower())
            x = activation_fn(x)
        y_pred = nn.Dense(self.n_output)(x)
        if self.predict_errors:
            y_pred, y_var = np.split(y_pred, 2, axis=-1)
            y_var = nn.softplus(y_var)
        else:
            y_var = jnp.zeros_like(y_pred)
        y_pred = y_pred * std_output + mean_output
        if self.transform_output is not None:
            y_pred = self.transform_output.inverse_transform(y_pred)
        if filters is not None:
            y_pred = y_pred[~filters.reshape(-1)]
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
            if 'mean' in key or 'std' in key:
                # Convert PyTorch tensors directly to numpy arrays without transposition
                jax_state[key] = np.array(tensor)
            elif "mlp" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                key = key.replace("mlp.mlp", f"Dense_")
                key = key.replace("mlp.act", f"FlaxLearnedSigmoid_")
                jax_state[key] = tensor.T
        return jax_state
