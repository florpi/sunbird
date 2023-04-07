import torch
import numpy as np
import json
from sunbird.covariance import CovarianceMatrix

from sunbird.summaries import Bundle
from sunbird.read_utils.data_utils import Abacus


def get_emulator_and_truth(
    split,
    select_filters={
        "quintiles": [0, 1, 3, 4],
        "multipoles": [0, 1],
    },
):
    with open("../data/train_test_split.json") as f:
        train_test_split = json.load(f)
    if type(split) is list:
        cosmo_idx = []
        for s in split:
            cosmo_idx += train_test_split[s]
    else:
        cosmo_idx = train_test_split[split]
    return get_emulator_and_truth_for_cosmo(
        cosmo_idx=cosmo_idx,
        select_filters=select_filters,
    )


def get_emulator_and_truth_for_cosmo(
    cosmo_idx,
    select_filters={
        "quintiles": [0, 1, 3, 4],
        "multipoles": [0, 1],
    },
    slice_filters={"s": [0.7, 150.0]},
):
    abacus = Abacus(select_filters=select_filters, slice_filters=slice_filters)
    parameters, true_density_split_auto, true_density_split_cross = [], [], []
    for cosmo in cosmo_idx:
        parameters.append(
            abacus.get_all_parameters(
                cosmology=cosmo,
            )
        )
        true_density_split_auto.append(
            abacus.read_statistic(
                statistic="density_split_auto",
                cosmology=cosmo,
                phase=0,
            ).values
        )
        true_density_split_cross.append(
            abacus.read_statistic(
                statistic="density_split_cross",
                cosmology=cosmo,
                phase=0,
            ).values
        )
        # true_density_split_cross.append(
        #    read_statistic_abacus(
        #        statistic='density_split_cross',
        #        cosmology=cosmo,
        #        dataset=dataset,
        #        select_filters=select_filters,
        #    ).values
        # )
    true_density_split_auto = np.array(true_density_split_auto)
    true_density_split_auto = true_density_split_auto.reshape(
        (
            -1,
            np.prod(list(true_density_split_auto.shape[2:])),
        )
    )
    true_density_split_cross = np.array(true_density_split_cross)
    true_density_split_cross = true_density_split_cross.reshape(
        (
            -1,
            np.prod(list(true_density_split_cross.shape[2:])),
        )
    )
    true_density_split = np.hstack((true_density_split_auto, true_density_split_cross))
    parameters = np.array(parameters)
    parameters = torch.tensor(parameters, dtype=torch.float32).reshape(
        -1, parameters.shape[-1]
    )
    # 2) Get emulator predictions for set to plot using Bundle
    emulator = Bundle(
        summaries=["density_split_auto", "density_split_cross"],
    )
    s = np.unique(emulator.all_summaries["density_split_auto"].model.s)
    emulated_density_split = (
        emulator.forward(
            parameters,
            select_filters=select_filters,
        )
    )
    return s, true_density_split, emulated_density_split, parameters


def get_data_variance(
    statistic,
    select_filters={
        "quintiles": [0, 1, 3, 4],
        "multipoles": [0, 1],
    },
    slice_filters={"s": [0.7, 150.0]},
):
    covariance = CovarianceMatrix(
        statistics=[statistic],
        select_filters=select_filters,
        slice_filters=slice_filters,
        covariance_data_class='AbacusSmall'
    )
    covariance_data = covariance.get_covariance_data(
        apply_hartlap_correction=True,
        volume_scaling=8.,
    )
    return np.sqrt(np.diag(covariance_data))
