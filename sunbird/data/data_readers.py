import numpy as np
import json
from abc import ABC
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional, Callable
from sunbird.data.data_utils import convert_to_summary, normalize_data
from sunbird.data.transforms import BaseTransform

DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class DataReader(ABC):
    def __init__(
        self,
        data_path: Optional[Path] = DATA_PATH,
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
            "voids",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
        avg_los: bool = True,
    ):
        """Base class for data readers

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
            avg_los (bool, optional): whether to average along line of sight. Defaults to True.
        """
        self.data_path = data_path
        self.transforms = transforms
        self.statistics = statistics
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.avg_los = avg_los

    def get_file_path(
        self,
        dataset: str,
        statistic: str,
        suffix: str,
    ) -> Path:
        """Get file path where data is stored

        Args:
            dataset (str): dataset to read
            statistic (str): summary statistic to read, one of: density_split_auto, density_split_cross, tpcf
            suffix (str): suffix

        Raises:
            ValueError: if statstics is not known

        Returns:
            Path: path to file
        """
        if statistic == "density_split_auto":
            return (
                self.data_path
                / f"clustering/{dataset}/ds/gaussian/ds_auto_multipoles_zsplit_Rs10_{suffix}.npy"
            )
        elif statistic == "density_split_cross":
            return (
                self.data_path
                / f"clustering/{dataset}/ds/gaussian/ds_cross_multipoles_zsplit_Rs10_{suffix}.npy"
            )
        elif statistic == "tpcf":
            return (
                self.data_path
                / f"clustering/{dataset}/tpcf/tpcf_multipoles_{suffix}.npy"
            )
        elif statistic == "voids":
            return (
                self.data_path
                / f"clustering/{dataset}/voids/voids_multipoles_{suffix}.npy"
            )
        elif statistic == "density_pdf":
            return (
                self.data_path
                / f"clustering/{dataset}/ds/gaussian/density_pdf_zsplit_Rs10_{suffix}.npy"
            )
        raise ValueError(f"Invalid statistic {statistic}")

    def get_observation(
        self, select_from_coords: Optional[Dict] = None, **kwargs
    ) -> np.array:
        """Get observation from data

        Args:
            select_from_coords (Optional[Dict], optional): whether to select values from a specific coordinate.
            Defaults to None.

        Returns:
            np.array: flattened observation
        """
        observation = []
        for statistic in self.statistics:
            observed = self.read_statistic(
                statistic=statistic,
                **kwargs,
            )
            if select_from_coords is not None:
                observed = observed.sel(select_from_coords)
            if hasattr(observation, "realizations"):
                n_realizations = len(observation.realizations)
                observed = observed.values.reshape(n_realizations, -1)
            else:
                observed = observed.values.reshape(-1)
            observation.append(observed)
        return np.hstack(observation)

    def gather_summaries_for_covariance(
        self,
    ) -> np.array:
        """
        Gather summaries for covariance matrix estimation

        Returns:
            np.array: flattened summaries
        """
        summaries = []
        for statistic in self.statistics:
            summary = self.read_statistic(
                statistic=statistic,
            )
            summary = np.array(summary.values).reshape(
                (len(summary["realizations"]), -1)
            )
            summary = normalize_data(
                summary,
                self.normalization_dict,
                standarize=self.standarize,
                normalize=self.normalize,
            )
            summaries.append(summary)
        return np.hstack(summaries)

    def read_statistic(
        self, statistic: str, multiple_realizations: bool = True, **kwargs
    ) -> xr.DataArray:
        """Read summary statistic from data as a DataArray

        Args:
            statistic (str): summary statistic to read
            multiple_realizations (bool, optional):  whether there are multiples realizations stored in
            the same file. Defaults to True.

        Returns:
            xr.DataArray: data array
        """
        with open(self.data_path / f"coordinates/{statistic}.json", "r") as f:
            coords = json.load(f)
        dimensions = list(coords.keys())
        path_to_file = self.get_file_path(statistic=statistic, **kwargs)
        original_data = np.load(
            path_to_file,
            allow_pickle=True,
        ).item()
        if statistic == "density_pdf":
            data = np.array(original_data["hist"])
        else:
            data = np.asarray(original_data["multipoles"])
        if multiple_realizations:
            dimensions.insert(0, "realizations")
            coords["realizations"] = np.arange(data.shape[0])
        if "multipoles" in original_data and self.avg_los:
            if multiple_realizations:
                data = np.mean(data, axis=1)
            else:
                data = np.mean(data, axis=0)
        summary = convert_to_summary(
            data=data,
            dimensions=dimensions,
            coords=coords,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )
        if self.transforms is not None and statistic in self.transforms.keys():
            summary = self.transforms[statistic].transform(summary)
        return summary


class Abacus(DataReader):
    def __init__(
        self,
        data_path: Optional[Path] = DATA_PATH,
        dataset: Optional[str] = "voidprior_AB",
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
            "voids",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """Abacus data class for the abacus summit latin hypercube of simulations.

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            dataset (str, optional): dataset to read. Defaults to "different_hods_linsigma".
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=True,
        )
        self.dataset = f"abacus/{dataset}"

    def get_file_path(
        self,
        statistic: str,
        cosmology: int,
        phase: int,
    ) -> Path:
        """get file path where data is stored for a given statistic, cosmology, and phase

        Args:
            statistic (str): summary statistic to read
            cosmology (int): cosmology to read within abacus' latin hypercube
            phase (int): phase to read

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset=self.dataset,
            statistic=statistic,
            suffix=f"c{str(cosmology).zfill(3)}_ph{str(phase).zfill(3)}",
        )

    def get_observation(
        self,
        cosmology: int,
        hod_idx: int,
        phase: int = 0,
    ) -> np.array:
        """get array of a given observation at a cosmology, hod_idx, and phase

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube
            hod_idx (int): hod_idx to read within a given cosmology
            phase (int): phase to read

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            cosmology=cosmology,
            phase=phase,
            select_from_coords={"realizations": hod_idx},
        )

    def get_all_parameters(self, cosmology: int) -> pd.DataFrame:
        """dataframe of parameters used for a given cosmology

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube

        Returns:
            pd.DataFrame: dataframe of cosmology + HOD parameters
        """
        return pd.read_csv(
            self.data_path
            / f"parameters/{self.dataset}/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        )

    def get_parameters_for_observation(
        self,
        cosmology: int,
        hod_idx: int,
    ) -> Dict:
        """get cosmology + HOD parameters for a particular observation

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube
            hod_idx (int): hod_idx to read within a given cosmology

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return self.get_all_parameters(cosmology=cosmology).iloc[hod_idx].to_dict()

    @property
    def cosmological_parameters(
        self,
    ):
        return [
            "omega_b",
            "omega_cdm",
            "sigma8_m",
            "n_s",
            "nrun",
            "N_ur",
            "w0_fld",
            "wa_fld",
        ]

class AbacusCutSky(DataReader):
    def __init__(
        self,
        data_path: Optional[Path] = DATA_PATH,
        dataset: Optional[str] = "voidprior",
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
            "voids",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """Abacus data class for the abacus summit latin hypercube of simulations.

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            dataset (str, optional): dataset to read. Defaults to "different_hods_linsigma".
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=False,
        )
        self.dataset = f"abacus_cutsky/{dataset}"

    def get_file_path(
        self,
        statistic: str,
        cosmology: int,
        phase: int,
    ) -> Path:
        """get file path where data is stored for a given statistic, cosmology, and phase

        Args:
            statistic (str): summary statistic to read
            cosmology (int): cosmology to read within abacus' latin hypercube
            phase (int): phase to read

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset=self.dataset,
            statistic=statistic,
            suffix=f"c{str(cosmology).zfill(3)}_ph{str(phase).zfill(3)}",
        )

    def get_observation(
        self,
        cosmology: int,
        hod_idx: int,
        phase: int = 0,
    ) -> np.array:
        """get array of a given observation at a cosmology, hod_idx, and phase

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube
            hod_idx (int): hod_idx to read within a given cosmology
            phase (int): phase to read

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            cosmology=cosmology,
            phase=phase,
            select_from_coords={"realizations": hod_idx},
        )

    def get_all_parameters(self, cosmology: int) -> pd.DataFrame:
        """dataframe of parameters used for a given cosmology

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube

        Returns:
            pd.DataFrame: dataframe of cosmology + HOD parameters
        """
        return pd.read_csv(
            self.data_path
            / f"parameters/{self.dataset}/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        )

    def get_parameters_for_observation(
        self,
        cosmology: int,
        hod_idx: int,
    ) -> Dict:
        """get cosmology + HOD parameters for a particular observation

        Args:
            cosmology (int): cosmology to read within abacus' latin hypercube
            hod_idx (int): hod_idx to read within a given cosmology

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return self.get_all_parameters(cosmology=cosmology).iloc[hod_idx].to_dict()

    @property
    def cosmological_parameters(
        self,
    ):
        return [
            "omega_b",
            "omega_cdm",
            "sigma8_m",
            "n_s",
            "nrun",
            "N_ur",
            "w0_fld",
            "wa_fld",
        ]


class AbacusSmall(DataReader):
    def __init__(
        self,
        dataset: str = "voidprior_AB",
        data_path: Optional[Path] = DATA_PATH,
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
            "voids",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        normalization_dict: Optional[Dict] = None,
        standarize: bool = False,
        normalize: bool = False,
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=True
            if dataset in ("widreprior_AB", "fixed_cosmo_bossprior")
            else False,
        )
        self.dataset = f"abacus_small/{dataset}"
        self.normalization_dict = normalization_dict
        self.standarize = standarize
        self.normalize = normalize

    def get_file_path(
        self,
        statistic: str,
    ):
        """get file path where data is stored for a given statistic

        Args:
            statistic (str): summary statistic to read

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset=self.dataset,
            statistic=statistic,
            suffix="c000_hod26",
        )

    def get_observation(
        self,
        phase: int,
    ) -> np.array:
        """get array of a given observation at a given phase

        Args:
            phase (int): random phase to read

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            select_from_coords={"realizations": phase},
            multiple_realizations=True,
        )

    def get_parameters_for_observation(
        self,
    ) -> Dict:
        """get cosmological parameters for a particular observation

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return {
            "omega_b": 0.02213,
            "omega_cdm": 0.11891,
            "sigma8_m": 0.8288,
            "n_s": 0.9611,
            "nrun": 0.0,
            "N_ur": 2.0328,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
        }


class Uchuu(DataReader):
    def __init__(
        self,
        data_path: Optional[Path] = DATA_PATH,
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """Uchuu data class to read Uchuu results

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            dataset (str, optional): dataset to read. Defaults to "different_hods_linsigma".
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=True,
        )

    def get_file_path(
        self,
        statistic: str,
        ranking: str,
    ) -> Path:
        """get file path where data is stored for a given statistic, cosmology, and phase

        Args:
            statistic (str): summary statistic to read
            ranking (str): whether to use ranked halos or random halos

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset="uchuu",
            statistic=statistic,
            suffix=f"{str(ranking)}",
        )

    def get_observation(
        self,
        ranking: str,
    ) -> np.array:
        """get array of a given observation at a cosmology and ranking

        Args:
            ranking (str): whether to use ranked halos or random halos

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            ranking=ranking,
            multiple_realizations=False,
        )

    def get_parameters_for_observation(
        self,
    ) -> Dict:
        """get cosmological parameters for a particular observation

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return {
            "omega_b": 0.02213,
            "omega_cdm": 0.11891,
            "sigma8_m": 0.8288,
            "n_s": 0.9611,
            "nrun": 0.0,
            "N_ur": 2.0328,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
        }


class Patchy(DataReader):
    def __init__(
        self,
        data_path: Optional[Path] = DATA_PATH,
        statistics: Optional[List[str]] = [
            "density_split_auto",
            "density_split_cross",
            "tpcf",
            "voids",
        ],
        select_filters: Optional[Dict] = {
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Optional[Dict] = {"s": [0.7, 150.0]},
        normalization_dict: Optional[Dict] = None,
        standarize: bool = False,
        normalize: bool = False,
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """Patchy data class for the pathcy mocks.

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=False,
        )
        self.normalization_dict = normalization_dict
        self.standarize = standarize
        self.normalize = normalize

    def get_file_path(
        self,
        statistic: str,
    ):
        """get file path where data is stored for a given statistic

        Args:
            statistic (str): summary statistic to read

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset="patchy",
            statistic=statistic,
            suffix=f"ngc_landyszalay",
        )

    def get_observation(
        self,
        phase: int,
    ) -> np.array:
        """get array of a given observation at a given phase

        Args:
            phase (int): random phase to read

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            select_from_coords={"realizations": phase},
            multiple_realizations=True,
        )

    def get_parameters_for_observation(
        self,
    ) -> Dict:
        """get cosmological parameters for a particular observation

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return {
            "omega_b": 0.02213,
            "omega_cdm": 0.11891,
            "sigma8_m": 0.8288,
            "n_s": 0.9611,
            "nrun": 0.0,
            "N_ur": 2.0328,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
        }


class CMASS(DataReader):
    def __init__(
        self,
        data_path=DATA_PATH,
        statistics: List[str] = ["density_split_auto", "density_split_cross", "tpcf", "voids"],
        select_filters: Dict = {
            "multipoles": [
                0,
                2,
            ],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Dict = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """CMASS data class to read CMASS data

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=True,
        )

    def get_file_path(
        self,
        statistic: str,
    ) -> Path:
        """get file path where data is stored for a given statistic, cosmology, and phase

        Args:
            statistic (str): summary statistic to read

        Returns:
            Path: path to where data is stored
        """
        return super().get_file_path(
            dataset="cmass",
            statistic=statistic,
            suffix="ngc_landyszalay",
        )

    def get_observation(
        self,
        galactic_cap="ngc",
    ) -> np.array:
        """get array of a given observation at a cosmology and ranking

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            multiple_realizations=False,
        )

    def get_parameters_for_observation(
        self,
        galactic_cap="ngc",
    ) -> Dict:
        """get cosmological parameters for a particular observation

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return {
            "omega_b": 0.02213,
            "omega_cdm": 0.11891,
            "sigma8_m": 0.8288,
            "n_s": 0.9611,
            "nrun": 0.0,
            "N_ur": 2.0328,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
        }


class Beyond2pt(DataReader):
    def __init__(
        self,
        data_path=DATA_PATH,
        statistics: List[str] = ["density_split_auto", "density_split_cross", "tpcf"],
        select_filters: Dict = {
            "multipoles": [
                0,
                2,
            ],
            "quintiles": [0, 1, 3, 4],
        },
        slice_filters: Dict = {"s": [0.7, 150.0]},
        transforms: Optional[Dict[str, BaseTransform]] = None,
    ):
        """data class to read Beyond2pt challenge data

        Args:
            data_path (Path, optional): path where data is stored. Defaults to DATA_PATH.
            statistics (List[str], optional): summary statistics to read.
            Defaults to ["density_split_auto", "density_split_cross", "tpcf"].
            select_filters (Dict, optional): filters to select values along coordinates.
            Defaults to {"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]}.
            slice_filters (Dict, optional): filters to slice values along coordinates.
            Defaults to {"s": [0.7, 150.0]}.
            transforms (Dict, optional): transforms to apply to data. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            transforms=transforms,
            avg_los=True,
        )

    def get_file_path(
        self,
        statistic: str,
    ) -> Path:
        """get file path where data is stored for a given statistic, cosmology, and phase

        Args:
            statistic (str): summary statistic to read

        Returns:
            Path: path to where data is stored
        """
        if statistic == "density_split_auto":
            return (
                self.data_path
                / f"clustering/beyond2pt/ds/gaussian/ds_auto_zsplit_Rs10_lcdm_redshift_space.npy"
            )
        elif statistic == "density_split_cross":
            return (
                self.data_path
                / f"clustering/beyond2pt/ds/gaussian/ds_cross_zsplit_Rs10_lcdm_redshift_space.npy"
            )
        elif statistic == "tpcf":
            return (
                self.data_path
                / f"clustering/beyond2pt/tpcf/tpcf_lcdm_redshift_space.npy"
            )
        raise ValueError(f"Invalid statistic {statistic}")

    def get_observation(
        self,
    ) -> np.array:
        """get array of a given observation at a cosmology and ranking

        Returns:
            np.array: flattened observation
        """
        return super().get_observation(
            multiple_realizations=False,
        )

    def get_parameters_for_observation(
        self,
    ) -> Dict:
        """Assume a fiducial model for when fixing parameters

        Returns:
            Dict: dictionary of cosmology + HOD parameters
        """
        return {
            "omega_b": 0.02237,
            "omega_cdm": 0.1200,
            "sigma8_m": 0.749999,
            "n_s": 0.9649,
            "nrun": 0.0,
            "N_ur": 2.0328,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
            "B_cen": 0.0,
            "B_sat": 0.0,
            "alpha_s": 0.0,
            "alpha_c": 0.0,
        }
