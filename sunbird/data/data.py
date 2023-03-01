from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import torch
import json
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sunbird.read_utils.data_utils import Abacus, normalize_data


class DSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_test_split_dict: Dict[str, int],
        statistic: str = "density_split_cross",
        select_filters: Optional[Dict] = None,
        slice_filters: Optional[Dict] = {'s': [0.7, 150.]},
        batch_size: int = 256,
        standarize_outputs: bool = False,
        normalize_outputs: bool = True,
        normalize_inputs: bool = True,
        s2_outputs: bool = False,
        abacus_dataset: Optional[str] = "wideprior_AB",
        inputs_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Data module used to train models on Abacus data

        Args:
            train_test_split_dict (Dict[str, int]): dictionary with train, test and val splits idx
            statistic (str, optional): statistic to fit. Defaults to "density_split_cross".
            select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
            slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.
            batch_size (int, optional): size of batch. Defaults to 32.
            standarize_outputs (bool, optional):  whether to standarize outputs. Defaults to False.
            normalize_outputs (bool, optional): whether to normalize outputs. Defaults to False.
            normalize_inputs (bool, optional): whether to normalize inputs. Defaults to True.
            s2_outputs (bool, optional): whether to multiply outputs by s squared (s is pair separation, useful
            on large scales). Defaults to False.
            abacus_dataset (Optional[str], optional): what abacus dataset to use. Defaults to 'wideprior_AB'.
            inputs_names (Optional[List[str]], optional): names of parameters to use as inputs, if None it will
            use all cosmology + HOD parameters. Defaults to None.
        """
        super().__init__()
        self.train_test_split_dict = train_test_split_dict
        self.statistic = statistic
        self.batch_size = batch_size
        self.standarize_outputs = standarize_outputs
        self.normalize_outputs = normalize_outputs
        self.normalize_inputs = normalize_inputs
        self.s2_outputs = s2_outputs
        self.data = Abacus(
            dataset=abacus_dataset,
            statistics=[self.statistic],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        self.inputs_names = inputs_names

    @classmethod
    def add_argparse_args(cls, parser):
        """Add arguments to parse

        Args:
            parser (parser): parser 

        Returns:
            parser: updated parser with args and defaults 
        """
        parser.add_argument('--statistic', type=str, default='density_split_cross',)
        parser.add_argument('--select_quintiles',  action='store', type=int, default=[0,],nargs='+',)
        parser.add_argument('--select_multipoles', action='store', type=int, default=[0,1],nargs='+',)
        parser.add_argument('--slice_s', action='store', type=float, default=[0.7,150.],nargs='+',)
        parser.add_argument('--batch_size', type=int, default=256,)
        parser.add_argument('--standarize_outputs', type=bool, default=False,)
        parser.add_argument('--normalize_outputs', type=bool, default=True,)
        parser.add_argument('--normalize_inputs', type=bool, default=True,)
        parser.add_argument('--s2_outputs', type=bool, default=False,)
        parser.add_argument('--abacus_dataset', type=str, default='wideprior_AB',)
        parser.add_argument('--input_names', action='store', type=str, default=None,nargs='+',)
        return parser

    @classmethod
    def from_argparse_args(cls, args, train_test_split_dict: Dict)->"DSDataModule":
        """ Create data module from parsed args

        Args:
            args (args): command line arguments 
            train_test_split_dict (Dict): train test split dictionary 

        Returns:
            DSDataModule: data module 
        """
        vargs = vars(args)
        select_filters, slice_filters = {}, {}
        for key, value in vargs.items():
            if 'select' in key:
                key_to_filter = key.split('_')[-1]
                if key_to_filter != 'gpu':
                    select_filters[key_to_filter] = value
            elif 'slice' in key:
                slice_filters[key.split('_')[-1]] = value
        return cls(
            train_test_split_dict=train_test_split_dict,
            select_filters=select_filters,
            slice_filters= slice_filters,
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
            if self.s2_outputs:
                summary = summary * summary.s**2
            n_realizations = len(summary.realizations)
            data += list(summary.values.reshape((n_realizations, -1)))
        return np.array(data)

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
            if self.inputs_names is not None:
                params_df = params_df[self.inputs_names]
            params += list(params_df.values)
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
        cosmology_idx = self.train_test_split_dict[stage]
        params = self.load_params(
            cosmology_idx=cosmology_idx,
        )
        data = self.load_data(
            cosmology_idx=cosmology_idx,
        )
        return params, data

    def summarise_training_data(
        self, parameters: np.array, data: np.array
    ) -> Dict[str, np.array]:
        """Summarise the trianing data, used to normalize inputs and outputs

        Args:
            parameters (np.array): input parameters
            data (np.array): data to fit

        Returns:
            Dict: dictionary with stats
        """
        return {
            "x_min": np.min(parameters, axis=0),
            "x_max": np.max(parameters, axis=0),
            "y_min": np.min(data),
            "y_max": np.max(data),
            "y_mean": np.mean(data, axis=0),
            "y_std": np.std(data, axis=0),
        }



    def generate_dataset(self, x: np.array, y: np.array) -> TensorDataset:
        """Convert numpy arrays to torch tensors and generate a normalized TensorDataset

        Args:
            x (np.array): array of inputs
            y (np.array): array of outputs

        Returns:
            TensorDataset: dataset
        """
        #TODO: homogeneize normalization in class
        x = normalize_data(
            data=x,
            normalization_dict=self.normalization_dict,
            standarize=False,
            normalize=self.normalize_inputs,
            coord='x',
        )
        y = normalize_data(
            data=y,
            normalization_dict=self.normalization_dict,
            standarize=self.standarize_outputs,
            normalize=self.normalize_outputs,
            coord='y',
        )
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
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
            self.normalization_dict = self.summarise_training_data(
                parameters=train_params,
                data=train_data,
            )
            self.ds_train = self.generate_dataset(
                train_params,
                train_data,
            )
        if not stage or stage == "train" or stage == "fit":
            val_params, val_data = self.load_params_and_data_for_stage(
                stage="val",
            )
            self.ds_val = self.generate_dataset(
                val_params,
                val_data,
            )

        if not stage or stage == "test":
            test_params, test_data = self.load_params_and_data_for_stage(
                stage="test",
            )
            self.ds_test = self.generate_dataset(
                test_params,
                test_data,
            )
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

    def dump_summaries(self, path: Path):
        """Store the summary of the training data,
        used to normalize inputs and outputs

        Args:
            path (Path): path to store data
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(path, "w") as fd:
            json_dump = json.dumps(self.normalization_dict, cls=NumpyEncoder)
            json.dump(json_dump, fd)

def load_summary_training(data_dir, statistic,):
    with open(
        data_dir / f"train_{statistic}_summary.json",
        "r",
    ) as f:
        summary = json.load(f)
    for k, v in summary.items():
        if type(v) is list:
            summary[k] = np.array(v)
    return summary
