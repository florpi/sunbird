from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import json
import yaml
import jax
import numpy as np
import jax.numpy as jnp
import xarray
import torch
import flax
from flax.core.frozen_dict import freeze

from sunbird.emulators import FCN, FlaxFCN
from sunbird.data import transforms
from sunbird.data.data_utils import convert_selection_to_filters, convert_to_summary

DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"

DEFAULT_COSMO_PARAMS = [
    "omega_b",
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "nrun",
    "N_ur",
    "w0_fld",
    "wa_fld",
]

DEFAULT_GAL_PARAMS = [
    "logM_cut",
    "logM1",
    "logsigma",
    "alpha",
    "kappa",
    "alpha_c",
    "alpha_s",
]


class BaseSummary:
    def __init__(
        self,
        model: Union[torch.nn.Module, flax.linen.Module],
        coordinates: Dict[str, np.array],
        input_transforms: Optional[transforms.Transforms] = None,
        output_transforms: Optional[transforms.Transforms] = None,
        flax_params: Optional[jnp.array] = None,
        input_names: Optional[List] = DEFAULT_COSMO_PARAMS + DEFAULT_GAL_PARAMS,
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
    def load_model(
        cls, path_to_model: Path, flax: bool
    ) -> Tuple[Union[torch.nn.Module, flax.linen.Module], Optional[jnp.array]]:
        """Load model from folder, either a torch or flax model

        Args:
            path_to_model (Path): path to model folder
            flax (bool): whether to use flax, if False it will load a pytorch model.

        Returns:
            Tuple[Union[torch.nn.Module, flax.linen.Module], Optional[jnp.array]]: model and flax parameters
        """

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
            if key in coordinates:
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
        select_filters: Optional[Dict] = None,
        slice_filters: Optional[Dict] = None,
        use_xarray: bool = False,
        batch: bool = False,
        return_errors: bool = True,
    ) -> Union[np.array, xarray.DataArray]:
        """Forward pass of the neural network

        Args:
            inputs (np.array): input parameters
            select_filters (Optional[Dict], optional): Filters used to select coordinates. Defaults to None.
            slice_filters (Optional[Dict], optional): Filters used to slice coordinates. Defaults to None.
            batch (bool, optional): whether to run a batch of parameters. Defaults to False.
            use_xarray (bool, optional): if True, returns an xarray object with coordinates. Defaults to False.

        Returns:
            prediction
        """
        if self.input_transforms is not None:
            inputs = self.input_transforms.transform(inputs)
        if self.flax:
            prediction, variance = self.model_apply(self.flax_params, inputs)
            errors = jnp.sqrt(variance)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction, variance = self.model(inputs)
            prediction = prediction.detach()
            errors = torch.sqrt(variance).detach()
        if self.output_transforms is not None:
            prediction, errors = self.apply_output_transforms(
                prediction, errors, batch=batch
            )
        prediction = self.apply_filters(
            prediction=prediction,
            select_filters=select_filters,
            slice_filters=slice_filters,
            batch=batch,
            use_xarray=use_xarray,
        )
        errors = self.apply_filters(
            prediction=errors,
            select_filters=select_filters,
            slice_filters=slice_filters,
            batch=batch,
            use_xarray=use_xarray,
        )
        if use_xarray:
            if return_errors:
                return prediction, errors
            return prediction
        if return_errors:
            return prediction.reshape(-1), errors.reshape(-1)
        return prediction.reshape(-1)

    def find_index(self, arr, num, mode="below"):
        if mode == "below":
            indices = np.where(arr < num)
        elif mode == "above":
            indices = np.where(arr > num)
        return indices[0][np.argmin(np.abs(num - arr[indices]))]

    def apply_select_filters(self, prediction, dimensions, coordinates, select_filters):
        idx_list = []
        for dim in range(prediction.ndim):
            key = dimensions[dim]
            if key in select_filters:
                idx_list.append(np.isin(coordinates[key], select_filters[key]))
            else:
                idx_list.append(np.ones(prediction.shape[dim], dtype=bool))
        full_idx = np.ix_(*idx_list)
        return prediction[full_idx]

    def apply_slice_filters(self, prediction, dimensions, coordinates, slice_filters):
        full_slice = [slice(None)] * prediction.ndim
        for key, (min_value, max_value) in slice_filters.items():
            if key in dimensions:
                key_idx = dimensions.index(key)
                min_value_idx = self.find_index(
                    coordinates[key], min_value, mode="above"
                )
                max_value_idx = self.find_index(
                    coordinates[key], max_value, mode="below"
                )
                full_slice[key_idx] = slice(min_value_idx, max_value_idx + 1)
        return prediction[tuple(full_slice)]

    def apply_filters(
        self,
        prediction: xarray.DataArray,
        select_filters: Dict,
        slice_filters: Dict,
        batch: bool,
        use_xarray: bool = False,
    ) -> xarray.DataArray:
        """Apply filters to prediction, based on coordinates

        Args:
            prediction (xarray.DataArray): prediction from neural network
            select_filters (Dict): select certain values in coordinates
            slice_filters (Dict): slice values in coordinates
            batch (bool): whether to run a batch of parameters

        Returns:
            xarray.DataArray: prediction with filters applied, includes coordinates
        """
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
        if use_xarray:
            if type(data) is torch.Tensor:
                data = data.detach().numpy()
            return convert_to_summary(
                data=data,
                dimensions=dimensions,
                coords=coordinates,
                select_filters=select_filters,
                slice_filters=slice_filters,
            )
        else:
            if select_filters is not None:
                data = self.apply_select_filters(
                    data, dimensions, coordinates, select_filters
                )
            if slice_filters is not None:
                data = self.apply_slice_filters(
                    data, dimensions, coordinates, slice_filters
                )
            return data

    def apply_output_transforms(
        self,
        prediction,
        predicted_errors,
        batch: bool = False,
    ):
        if batch:
            coordinates = self.coordinates.copy()
            coordinates["batch"] = range(len(prediction))
            prediction = prediction.reshape(
                (
                    len(prediction),
                    *self.coordinates_shape,
                )
            )
            predicted_errors = predicted_errors.reshape(
                (
                    len(prediction),
                    *self.coordinates_shape,
                )
            )
        else:
            coordinates = self.coordinates
            prediction = prediction.reshape(self.coordinates_shape)
            predicted_errors = predicted_errors.reshape(self.coordinates_shape)
        dimensions = list(self.coordinates.keys())
        return self.output_transforms.inverse_transform(
            prediction, predicted_errors, summary_dimensions=dimensions, batch=batch
        )

    def __call__(
        self,
        param_dict: Dict,
        select_filters: Optional[Dict] = None,
        slice_filters: Optional[Dict] = None,
        use_xarray: bool = False,
        return_errors: bool = True,
    ) -> Union[np.array, xarray.DataArray]:
        inputs = np.array([param_dict[k] for k in self.input_names]).reshape(1, -1)
        return self.get_for_sample(
            inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            use_xarray=use_xarray,
            return_errors=return_errors,
        )

    def get_for_sample(self, inputs, select_filters, slice_filters, use_xarray, return_errors):
        return self.forward(
            inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            use_xarray=use_xarray,
            return_errors=return_errors,
        )

    def get_for_batch(
        self,
        param_dict,
        select_filters,
        slice_filters,
        use_xarray=False,
        return_errors=True,
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.input_names]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(
            inputs=inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            use_xarray=use_xarray,
            return_errors=return_errors,
        )

    def get_for_batch_inputs(
        self,
        inputs,
        select_filters=None,
        slice_filters=None,
        use_xarray=False,
        return_errors=True,
    ):
        outputs, variance = self.forward(
            inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
            batch=True,
            use_xarray=use_xarray,
            return_errors=return_errors,
        )
        if use_xarray:
            outputs = outputs.values
            variance = variance.values
        return (
            outputs.reshape((len(inputs), -1)),
            variance.reshape((len(inputs), -1)),
        )


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
        path_to_models = Path(path_to_models)
        if n_hod_realizations is not None:
            path_to_model = (
                path_to_models
                / f"{dataset}/{loss}/{statistic}_hod{n_hod_realizations}"
            )
        else:
            path_to_model = path_to_models / f"{dataset}/{loss}/{statistic}"
        if suffix is not None:
            path_to_model = path_to_model.parent / (path_to_model.name + f"_{suffix}")
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
        fixed_cosmology = config["fixed_cosmology"]
        if fixed_cosmology is not None:
            input_names = DEFAULT_GAL_PARAMS
        else:
            input_names = DEFAULT_COSMO_PARAMS + DEFAULT_GAL_PARAMS
        super().__init__(
            model=model,
            coordinates=coordinates,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            flax_params=flax_params,
            input_names=input_names,
        )
