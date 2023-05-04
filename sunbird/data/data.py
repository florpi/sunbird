from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import torch
import json
import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sunbird.data.data_readers import Abacus
from sunbird.data.data_utils import convert_selection_to_filters
from sunbird.data import transforms

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data/"


class AbacusDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_test_split_dict: Dict[str, int],
        statistic: str = "density_split_cross",
        select_filters: Optional[Dict] = None,
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        batch_size: int = 256,
        abacus_dataset: Optional[str] = "wideprior_AB",
        input_parameters: Optional[List[str]] = None,
        input_transforms: Optional[transforms.Transforms] = transforms.Transforms(
            [transforms.Normalize(dimensions=(0,))]
        ),
        output_transforms: Optional[transforms.Transforms] = transforms.Transforms(
            [
                transforms.Normalize(
                    dimensions=[
                        "cosmology",
                        "realizations",
                        "quintiles",
                        "s",
                    ]
                )
            ]
        ),
        n_hod_realizations: Optional[int] = None,
        fixed_cosmology: Optional[int] = None,
        **kwargs,
    ):
        """Data module used to train models on Abacus data

        Args:
            train_test_split_dict (Dict[str, int]): dictionary with train, test and val splits idx
            statistic (str, optional): statistic to fit. Defaults to "density_split_cross".
            select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
            slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.
            batch_size (int, optional): size of batch. Defaults to 32.
            abacus_dataset (Optional[str], optional): what abacus dataset to use. Defaults to 'wideprior_AB'.
            input_parameters (Optional[List[str]], optional): names of parameters to use as inputs, if None it will
            use all cosmology + HOD parameters. Defaults to None.
            output_transforms (str, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__()
        self.train_test_split_dict = train_test_split_dict
        self.statistic = statistic
        self.batch_size = batch_size
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.fixed_cosmology = fixed_cosmology
        self.data = Abacus(
            dataset=abacus_dataset,
            statistics=[self.statistic],
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )
        self.input_parameters = input_parameters
        self.n_hod_realizations = n_hod_realizations

    @classmethod
    def add_argparse_args(cls, parser):
        """Add arguments to parse

        Args:
            parser (parser): parser

        Returns:
            parser: updated parser with args and defaults
        """
        parser.add_argument(
            "--statistic",
            type=str,
            default="density_split_cross",
        )
        parser.add_argument(
            "--select_quintiles",
            action="store",
            type=int,
            default=[
                0,
            ],
            nargs="+",
        )
        parser.add_argument(
            "--select_multipoles",
            action="store",
            type=int,
            default=[0, 2],
            nargs="+",
        )
        parser.add_argument(
            "--slice_s",
            action="store",
            type=float,
            default=[0.7, 150.0],
            nargs="+",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
        )
        parser.add_argument(
            "--abacus_dataset",
            type=str,
            default="wideprior_AB",
        )
        parser.add_argument(
            "--input_parameters",
            action="store",
            type=str,
            default=None,
            nargs="+",
        )
        parser.add_argument(
            "--independent_avg_scale",
            action="store",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--fixed_cosmology",
            action="store",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--n_hod_realizations",
            action="store",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--input_transforms",
            action="store",
            type=str,
            default=['Normalize'],
            nargs="+",
        )
        parser.add_argument(
            "--output_transforms",
            action="store",
            type=str,
            default=['Normalize'],
            nargs="+",
        )
        return parser

    @classmethod
    def from_argparse_args(
        cls,
        args,
        train_test_split_dict: Dict,
        data_dir: Path = DEFAULT_DATA_DIR,
    ) -> "AbacusDataModule":
        """Create data module from parsed args

        Args:
            args (args): command line arguments
            train_test_split_dict (Dict): train test split dictionary

        Returns:
            DSDataModule: data module
        """
        vargs = vars(args)
        select_filters, slice_filters = convert_selection_to_filters(vargs)
        if vargs['input_transforms'] is not None:
            input_transforms = transforms.Transforms(
                [getattr(transforms, transform)(dimensions=(0,)) for transform in vargs['input_transforms']]
            )
        else:
            input_transforms = None
        if vargs['output_transforms'] is not None:
            dimensions_to_exclude_from_average = list(select_filters.keys())
            if vargs["independent_avg_scale"]:
                dimensions_to_exclude_from_average += ["s"]
            with open(data_dir / f'coordinates/{vargs["statistic"]}.json') as f:
                dims = list(json.load(f).keys())
            dims += ['cosmology', 'realizations']
            dimensions = [dim for dim in dims if dim not in dimensions_to_exclude_from_average]
            output_transforms = transforms.Transforms(
                [
                    getattr(transforms, transform)(dimensions=dimensions)
                    for transform in vargs['output_transforms']
                ]
            )
        else:
            output_transforms = None
        vargs = {
            k: v
            for k, v in vargs.items()
            if k not in ("input_transforms", "output_transforms")
        }
        return cls(
            train_test_split_dict=train_test_split_dict,
            select_filters=select_filters,
            slice_filters=slice_filters,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            **vargs,
        )

    def load_data(
        self,
        cosmology_idx: List[int],
    ) -> np.array:
        """Given a list of cosmology indices, load the data for those cosmologies
        (note that each cosmology contains a given number of HOD realizations)

        Args:
            cosmology_idx (List[int]): list of cosmology idx to load

        Returns:
            np.array: data array
        """
        data = []
        for cosmology in cosmology_idx:
            summary = self.data.read_statistic(
                statistic=self.statistic,
                cosmology=cosmology,
                phase=0,
            )
            if self.n_hod_realizations is not None:
                summary = summary.sel(
                    realizations=slice(self.n_hod_realizations-1)
                )
            data += [summary]
        return xr.concat(data, dim="cosmology")

    def load_params(
        self,
        cosmology_idx: List[int],
    ) -> np.array:
        """Given a list of cosmology indices, load the cosmology + HOD parameters for those cosmologies
        (note that each cosmology contains a given number of HOD realizations)

        Args:
            cosmology_idx (List[int]): list of cosmology idx to load

        Returns:
            np.array: data array
        """
        params = []
        for cosmology in cosmology_idx:
            params_df = self.data.get_all_parameters(
                cosmology=cosmology,
            )
            if self.input_parameters is not None:
                params_df = params_df[self.input_parameters]
            if self.n_hod_realizations is not None:
                params_values = params_df.values[:self.n_hod_realizations]
            else:
                params_values = params_df.values
            params += list(params_values)
        return np.array(params)

    def load_params_and_data_for_stage(
        self,
        stage: str,
    ) -> Tuple[np.array]:
        """Load data for a given stage (train, test or val)

        Args:
            stage (str): one of train, test or val

        Returns:
            np.array: params and data arrays
        """
        if self.fixed_cosmology is not None:
            return self.load_params_and_data_for_stage_for_fixed_cosmology(
                stage=stage,
            )
        return self.load_params_and_data_for_stage_vary_cosmology(
            stage=stage,
        )

    def load_params_and_data_for_stage_vary_cosmology(
        self,
        stage: str,
    ) -> Tuple[np.array]:
        """Load data for a given stage (train, test or val)

        Args:
            stage (str): one of train, test or val

        Returns:
            np.array: params and data arrays
        """
        cosmology_idx = self.train_test_split_dict[stage]
        params = self.load_params(
            cosmology_idx=cosmology_idx,
        )
        data = self.load_data(
            cosmology_idx=cosmology_idx,
        )
        return params, data

    def load_params_and_data_for_stage_for_fixed_cosmology(
        self,
        stage: str,
    ) -> Tuple[np.array]:
        """Load data for a given stage (train, test or val)

        Args:
            stage (str): one of train, test or val

        Returns:
            np.array: params and data arrays
        """
        cosmology_idx = [self.fixed_cosmology]
        params = self.load_params(
            cosmology_idx=cosmology_idx,
        )[0]
        params = params[self.train_test_split_dict[stage]]
        data = self.load_data(
            cosmology_idx=cosmology_idx,
        )[0]
        data = data[self.train_test_split_dict[stage]]
        return params, data

    def generate_dataset(
        self, x: np.array, y: np.array, stage: str = "train"
    ) -> TensorDataset:
        """Convert numpy arrays to torch tensors and generate a normalized TensorDataset

        Args:
            x (np.array): array of inputs
            y (np.array): array of outputs

        Returns:
            TensorDataset: dataset
        """
        if stage == "train":
            if self.input_transforms is not None:
                x = self.input_transforms.fit_transform(
                    x,
                )
            if self.output_transforms is not None:
                y = self.output_transforms.fit_transform(
                    y,
                )
        else:
            if self.input_transforms is not None:
                x = self.input_transforms.transform(x)
            if self.output_transforms is not None:
                y = self.output_transforms.transform(y)
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y.values.reshape((len(x), -1)), dtype=torch.float32),
        )

    def setup(self, stage: Optional[str] = None):
        """set up the train/test/val datasets

        Args:
            stage (Optional[str], optional): stage to set up. Defaults to None.
        """
        if not stage or stage == "train" or stage == "fit":
            train_params, train_data = self.load_params_and_data_for_stage(
                stage="train",
            )
            self.ds_train = self.generate_dataset(
                train_params,
                train_data,
                stage="train",
            )
        if not stage or stage == "train" or stage == "fit":
            val_params, val_data = self.load_params_and_data_for_stage(
                stage="val",
            )
            self.ds_val = self.generate_dataset(
                val_params,
                val_data,
                stage="val",
            )

        if not stage or stage == "test":
            test_params, test_data = self.load_params_and_data_for_stage(
                stage="test",
            )
            self.ds_test = self.generate_dataset(test_params, test_data, stage="test")
        try:
            self.n_output = self.ds_train.tensors[1].shape[-1]
            self.n_input = self.ds_train.tensors[0].shape[-1]
        except:
            self.n_output = self.ds_test.tensors[1].shape[-1]
            self.n_input = self.ds_test.tensors[0].shape[-1]

    def train_dataloader(self) -> DataLoader:
        """get train dataloader

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """get val dataloader

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """get test dataloader

        Returns:
            DataLoader: test dataloader
        """
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def store_transforms(self, path: Path):
        """Store transforms

        Args:
            path (Path): path to store data
        """
        self.input_transforms.store_transform_params(path.parent / (path.name + '_input.pkl'))
        self.output_transforms.store_transform_params(path.parent / (path.name + '_output.pkl'))
