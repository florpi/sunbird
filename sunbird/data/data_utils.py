import numpy as np
import xarray as xr
from typing import List, Dict, Optional, Tuple


def transform_filters_to_slices(filters: Dict) -> Dict:
    """Transform a dictionary of filters into slices that select from min to max

    Example:
        filters = {'r': (10,100)} , will select the summary statistics for 10 < r < 100

    Args:
        filters (Dict): dictionary of filters.
    Returns:
        Dict: dictionary of filters with slices
    """
    slice_filters = filters.copy()
    for filter, (min, max) in filters.items():
        slice_filters[filter] = slice(min, max)
    return slice_filters


def convert_to_summary(
    data: np.array,
    dimensions: List[str],
    coords: Dict,
    select_filters: Optional[Dict] = None,
    slice_filters: Optional[Dict] = None,
) -> xr.DataArray:
    """Convert numpy array to DataArray summary to filter and select from

    Example:
        slice_filters = {'s': (0, 0.5),}, select_filters = {'multipoles': (0, 2),}
        will return the summary statistics for 0 < s < 0.5 and multipoles 0 and 2

    Args:
        data (np.array): numpy array containing data
        dimensions (List[str]): dimensions names (need to have the same ordering as in data array)
        coords (Dict): coordinates for each dimension
        select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
        slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.

    Returns:
        xr.DataArray: data array summary
    """
    if select_filters:
        select_filters = {k: v for k, v in select_filters.items() if k in dimensions}
    if slice_filters:
        slice_filters = {k: v for k, v in slice_filters.items() if k in dimensions}
    summary = xr.DataArray(
        data,
        dims=dimensions,
        coords=coords,
    )
    if select_filters:
        summary = summary.sel(**select_filters)
    if slice_filters:
        slice_filters = transform_filters_to_slices(slice_filters)
        summary = summary.sel(**slice_filters)
    return summary


def normalize_data(
    data: np.array,
    normalization_dict: Dict,
    standarize: bool,
    normalize: bool,
    coord: str = "y",
) -> Tuple[np.array]:
    """normalize the data given the training data summary

    Args:
        data (np.array): data
        normalization_dict (Dict): dictionary with normalization parameters
        standarize (bool): if True, standarize the data
        normalize (bool): if True, normalize the data
        coord (str, optional): coordinate to normalize, either x for parameters or y for data. Defaults to 'y'.

    Returns:
        Tuple[np.array]: normalized parameters and data
    """
    if normalize:
        return (data - normalization_dict[f"{coord}_min"]) / (
            normalization_dict[f"{coord}_max"] - normalization_dict[f"{coord}_min"]
        )
    elif standarize:
        return (data - normalization_dict[f"{coord}_mean"]) / normalization_dict[
            f"{coord}_std"
        ]
    return data


def convert_selection_to_filters(selections: List[Dict]) -> Tuple[Dict, Dict]:
    """Given a list of selections, either to select values across a dimension
    or slice values across a dimension, convert them into valid xarray filters

    Args:
        selections: list of dictionaries with selections

    Returns:
        Tuple[Dict, Dict]: select and slice filters
    """
    select_filters, slice_filters = {}, {}
    for key, value in selections.items():
        if "select" in key:
            key_to_filter = key.split("_")[-1]
            if "gpu" not in key_to_filter:
                select_filters[key_to_filter] = value
        elif "slice" in key:
            slice_filters[key.split("_")[-1]] = value
    return select_filters, slice_filters
