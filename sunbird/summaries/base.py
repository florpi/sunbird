from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import json
import yaml
import jax
import numpy as np
import jax.numpy as jnp
import torch
import flax
from flax.core.frozen_dict import freeze

from sunbird.emulators import FCN, FlaxFCN
from sunbird.data import transforms
from sunbird.data.data_utils import convert_selection_to_filters, convert_to_summary

DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class BaseSummary:
    def __init__(
        self,
        model: Union[torch.nn.Module, flax.linen.Module],
        coordinates: Dict[str, np.array],
        input_transforms: Optional[transforms.Transforms] = None,
        output_transforms: Optional[transforms.Transforms] = None,
        flax_params: Optional[jnp.array] = None,
        input_names: Optional[List] = [
            "omega_b",
            "omega_cdm",
            "sigma8_m",
            "n_s",
            "nrun",
            "N_ur",
            "w0_fld",
            "wa_fld",
            "logM1",
            "logM_cut",
            "alpha",
            "alpha_s",
            "alpha_c",
            "logsigma",
            "kappa",
            "B_cen",
            "B_sat",
        ],
    ):
        """Base class for summary statistics emulators

        Args:
            model (Union[torch.nn.Module, flax.linen.Module]): either a torch or flax model
            coordinates (Dict[str, np.array]): dictionary of coordinates where the summary statistics is evaluated
            input_transforms (Optional[transforms.Transforms], optional): transforms to apply to the neural net inputs.
            Defaults to None.
            output_transforms (Optional[transforms.Transforms], optional): tranforms to apply to the neural net outputs.
            Defaults to None.
            flax_params (Optional[jnp.array], optional):  parameters of the flax model, if using flax.
            Defaults to None.
            input_names (Optional[List], optional): names for input parameters.
            Defaults to [ "omega_b", "omega_cdm", "sigma8_m", "n_s", "nrun", "N_ur", "w0_fld",
            "wa_fld", "logM1", "logM_cut", "alpha", "alpha_s", "alpha_c", "logsigma", "kappa", "B_cen", "B_sat", ].
        """
        self.model = model
        self.flax_params = flax_params
        if self.flax_params is not None:
            self.flax = True
            self.flax_params = freeze({"params": self.flax_params})
            self.model_apply = jax.jit(
                lambda params, inputs: model.apply(params, inputs)
            )

        else:
            self.flax = False
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.input_names = input_names
        self.coordinates = coordinates
        self.coordinates_shape = tuple(len(v) for k, v in coordinates.items())

    @classmethod
    def from_folder(
        cls,
        path_to_model: Path,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
    ) -> "BaseSummary":
        """Load a base summary from folder with trained neural network model

        Args:
            path_to_model (Path): path where model weights and transforms are stored
            path_to_data (Path, optional): path to data folder. Defaults to DEFAULT_DATA_PATH.
            flax (bool, optional): whether to use flax, if False it will load a pytorch model.
            Defaults to False.

        Returns:
            BaseSummary: summary
        """
        path_to_model = Path(path_to_model)
        model, flax_params = cls.load_model(
            path_to_model=path_to_model,
            flax=flax,
        )
        input_transforms, output_transforms = cls.load_transforms(
            path_to_model=path_to_model
        )
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        coordinates = cls.load_coordinates(config=config, path_to_data=path_to_data)
        return cls(
            model=model,
            coordinates=coordinates,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            flax_params=flax_params,
        )

    @classmethod
    def load_model(cls, path_to_model, flax: bool):
        if flax:
            nn_model, flax_params = FlaxFCN.from_folder(
                path_to_model,
            )
        else:
            nn_model = FCN.from_folder(
                path_to_model,
                load_loss=False,
            )
            nn_model.eval()
            flax_params = None
        return nn_model, flax_params

    @classmethod
    def load_coordinates(
        cls, config: Dict, path_to_data: Path = DEFAULT_DATA_PATH
    ) -> Dict[str, np.array]:
        """Load coordinates for summary statistic from json file

        Args:
            config (Dict): config used to train model
            path_to_data (Path, optional): path to data folder. Defaults to DEFAULT_DATA_PATH.

        Returns:
            Dict[str, np.array]: dictionary with coordinates
        """

        with open(path_to_data / f'coordinates/{config["statistic"]}.json') as fd:
            coordinates = json.load(fd)
        coordinates = {k: np.array(v) for k, v in coordinates.items()}
        select_filters, slice_filters = convert_selection_to_filters(config)
        for key, values in select_filters.items():
            if key in coordinates:
                coordinates[key] = [v for v in coordinates[key] if v in values]
        for key, values in slice_filters.items():
            min_value, max_value = slice_filters[key]
            coordinates[key] = coordinates[key][
                (coordinates[key] > min_value) & (coordinates[key] < max_value)
            ]
        return coordinates

    @classmethod
    def load_transforms(cls, path_to_model: Path) -> Tuple[transforms.Transforms]:
        """Load transforms used during training from folder

        Args:
            path_to_model (Path): path to model folder

        Returns:
            Tuple[transforms.Transforms]: input and output transforms
        """
        input_transforms = transforms.Transforms.from_file(
            path_to_model / "transforms_input.pkl"
        )
        output_transforms = transforms.Transforms.from_file(
            path_to_model / "transforms_output.pkl"
        )
        return input_transforms, output_transforms

    def forward(
        self,
        inputs: np.array,
        select_filters=None,
        slice_filters=None,
        batch: bool = False,
        return_xarray: bool = False,
    ):
        if self.input_transforms is not None:
            inputs = self.input_transforms.transform(inputs)
        if self.flax:
            prediction = self.model_apply(self.flax_params, inputs)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction = self.model(inputs)
        # Invert transform
        prediction = self.apply_filters(
            prediction=prediction,
            select_filters=select_filters,
            slice_filters=slice_filters,
            batch=batch,
        )
        if self.output_transforms is not None:
            prediction = self.output_transforms.inverse_transform(prediction)
        if return_xarray:
            return prediction
        return prediction.values.reshape(-1)

    def apply_filters(self, prediction, select_filters, slice_filters, batch):
        if batch:
            dimensions = ["batch"] + list(self.coordinates.keys())
            coordinates = self.coordinates.copy()
            coordinates["batch"] = range(len(prediction))
            data = prediction.reshape(
                (
                    len(prediction),
                    *self.coordinates_shape,
                )
            )
        else:
            dimensions = list(self.coordinates.keys())
            coordinates = self.coordinates
            data = prediction.reshape(self.coordinates_shape)
        if type(data) is torch.Tensor:
            data = data.detach().numpy()
        return convert_to_summary(
            data=data,
            dimensions=dimensions,
            coords=coordinates,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )

    def __call__(
        self,
        param_dict,
        select_filters=None,
        slice_filters=None,
        return_xarray: bool = False,
    ):
        inputs = np.array([param_dict[k] for k in self.input_names]).reshape(1, -1)
        output = self.get_for_sample(
            inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            return_xarray=return_xarray,
        )
        if return_xarray:
            return output
        return output.reshape(-1)

    def get_for_sample(self, inputs, select_filters, slice_filters, return_xarray):
        if self.flax:
            return self.forward(
                inputs,
                select_filters=select_filters,
                slice_filters=slice_filters,
                return_xarray=return_xarray,
            )
        else:
            return self.forward(
                inputs,
                select_filters=select_filters,
                slice_filters=slice_filters,
                return_xarray=return_xarray,
            )

    def get_for_batch(
        self,
        param_dict,
        select_filters,
        slice_filters,
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.input_names]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(
            inputs=inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )

    def get_for_batch_inputs(
        self,
        inputs,
        select_filters=None,
        slice_filters=None,
    ):
        outputs = self.forward(
            inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            batch=True,
        )
        return outputs.reshape((len(inputs), -1))


class BaseSummaryFolder(BaseSummary):
    def __init__(
        self,
        statistic: str,
        dataset: str,
        loss: str = "mae",
        n_hod_realizations: Optional[int] = None,
        suffix: Optional[str] = None,
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
        **kwargs,
    ):
        if n_hod_realizations is not None:
            path_to_model = (
                path_to_models
                / f"best/{dataset}/{loss}/{statistic}_hod{n_hod_realizations}"
            )
        else:
            path_to_model = path_to_models / f"best/{dataset}/{loss}/{statistic}"
        if suffix is not None:
            path_to_model = path_to_model.parent / (path_to_model.name + f"_{suffix}")
        print('path to model')
        print(path_to_model)
        model, flax_params = self.load_model(
            path_to_model=path_to_model,
            flax=flax,
        )
        input_transforms, output_transforms = self.load_transforms(
            path_to_model=path_to_model
        )
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        coordinates = self.load_coordinates(config=config, path_to_data=path_to_data)
        super().__init__(
            model=model,
            coordinates=coordinates,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            flax_params=flax_params,
        )
