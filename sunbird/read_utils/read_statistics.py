import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Dict

DATA_PATH = Path(__file__).parent.parent.parent / "data/"

def transform_filters_to_slices(filters: Dict)->Dict:
    """Transform a dictionary of filters into slices that select from min to max
    Args:
        filters (Dict): dictionary of filters. Example:
            filters = {'r': (10,100)} , will select the summary statistics for 10 < r < 100
    Returns:
        Dict: dictionary of filters with slices
    """
    slice_filters = filters.copy()
    for filter, (min, max) in filters.items():
        slice_filters[filter] = slice(min, max)
    return slice_filters

def convert_to_summary(
        data: np.array,
        dimensions: Dict,
        select_filters=None,
        slice_filters=None,
    )->xr.DataArray:
    """Convert numpy array to Dataarray summary

    Args:
        data (np.array): numpy array containing data 
        dimensions (Dict[str]): dimensions names and values 
        select_filters (_type_, optional): select filters. Defaults to None.
        slice_filters (_type_, optional): slice filters. Defaults to None.

    Returns:
        xr.DataArray: data array summary
    """
    summary = xr.DataArray(
        data,
        dims=list(dimensions.keys()),
        coords=dimensions,
    )
    if select_filters:
        summary = summary.sel(**select_filters)
    if slice_filters:
        slice_filters = transform_filters_to_slices(slice_filters)
        summary = summary.sel(**slice_filters)
    return summary

def read_ds_statistic(
    path_to_file, select_filters=None, slice_filters=None,  avg_los=False,
) :
    data = np.load(
        path_to_file,
        allow_pickle=True,
    ).item()
    s = data['s']
    data = data['multipoles']
    if avg_los:
        data = np.mean(data,axis=1)
    phases = list(range(len(data)))
    quintiles = list(range(5))
    multipoles = list(range(3))
    dimensions = {
        'phases': phases,
        'quintiles': quintiles,
        'multipoles': multipoles,
        's': s,
    }
    return convert_to_summary(
        data=data,
        dimensions=dimensions,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )

def read_tpcf_statistic(
    path_to_file, select_filters=None, slice_filters=None, avg_los=False
) :
    data = np.load(
        path_to_file,
        allow_pickle=True,
    ).item()
    s = data['s']
    data = data['multipoles']
    if avg_los:
        data = np.mean(data,axis=1)
    phases = list(range(len(data)))
    multipoles = list(range(3))
    dimensions = {
        'phases': phases,
        'multipoles': multipoles,
        's': s,
    }
    return convert_to_summary(
        data=data,
        dimensions=dimensions,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )


def read_statistics_for_covariance(
    statistic,
    select_filters=None,
    slice_filters=None,
):
    if statistic == 'density_split_auto':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"covariance/ds/gaussian/ds_auto_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=False,
    ) 
    elif statistic == 'density_split_cross':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"covariance/ds/gaussian/ds_cross_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=False,
    ) 
    elif statistic == 'tpcf':
        return read_tpcf_statistic(
            path_to_file = DATA_PATH / f"covariance/xi_smu/xi_smu_landyszalay_randomsX50.npy",
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
    else:
        raise ValueError(f'{statistic} is not implemented!')

def read_statistic_abacus(
        statistic,
        cosmology,
        dataset,
        select_filters=None,
        slice_filters=None,
    ):
    if statistic == 'density_split_auto':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"clustering/{dataset}/ds/gaussian/ds_auto_xi_smu_zsplit_Rs20_c{str(cosmology).zfill(3)}_ph000.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=True,
    ) 
    elif statistic == 'density_split_cross':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"clustering/{dataset}/ds/gaussian/ds_cross_xi_smu_zsplit_Rs20_c{str(cosmology).zfill(3)}_ph000.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=True,
    ) 
    elif statistic == 'tpcf':
        return read_tpcf_statistic(
            path_to_file = DATA_PATH / f"clustering/{dataset}/xi_smu/xi_smu_c{str(cosmology).zfill(3)}_ph000.npy",
            select_filters=select_filters,
            slice_filters=slice_filters,
            avg_los=True,
        )
    else:
        raise ValueError(f'{statistic} is not implemented!')

def read_statistic_patchy(
        statistic,
        select_filters=None,
        slice_filters=None,
    ):
    if statistic == 'density_split_auto':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"clustering/patchy/ds/gaussian/ds_auto_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=False,
    ) 
    elif statistic == 'density_split_cross':
       return read_ds_statistic(
        path_to_file = DATA_PATH / f"clustering/patchy/ds/gaussian/ds_cross_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        select_filters=select_filters,
        slice_filters=slice_filters,
        avg_los=False,
    ) 
    elif statistic == 'tpcf':
        return read_tpcf_statistic(
            path_to_file = DATA_PATH / f"clustering/patchy/xi_smu/xi_smu_landyszalay_randomsX50.npy",
            select_filters=select_filters,
            slice_filters=slice_filters,
            avg_los=False,
        )
    else:
        raise ValueError(f'{statistic} is not implemented!')


def read_parameters_abacus(cosmology: int, dataset: str):
    return pd.read_csv(
        DATA_PATH / f"parameters/{dataset}/AbacusSummit_c{str(cosmology).zfill(3)}_hod1000.csv"
    )

def read_parameters_patchy():
    return pd.read_csv(
        DATA_PATH / f"parameters/patchy/patchy.csv"
    )
