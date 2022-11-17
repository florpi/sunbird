from sunbird.summaries import DensitySplit, TPCF
import numpy as np
from pathlib import Path
import xarray as xr
import torch
from matplotlib import pyplot as plt

DATA_PATH = Path(__file__).parent.parent.parent / "data/covariance/"
TEST_COSMOS = list(range(5))


class CovarianceMatrix():
    def __init__(
        self,
        statistic,
        filters
    ):
        self.statistic = statistic
        self.filters = filters

    @classmethod
    def get_covariance_data(
        cls,
        statistic,
        filters,
    ) -> np.array:
        if statistic == 'density_split':
            data = np.load(
                DATA_PATH
                / "ds_cross_xi_smu_zsplit_Rs20_landyszalay_randomsX50.npy",
                allow_pickle=True,
            ).item()
            quintiles = range(5)
        else:
            raise ValueError(f'{statistic} is not implemented!')
        s = data["s"]
        data = data['multipoles']
        print(np.shape(data))
        if statistic == 'density_split':
            data = xr.DataArray(
                data, 
                dims=("phases", "quintiles", "multipoles", "s"), 
                coords={
                    "phases": list(range(len(data))),
                    "quintiles": list(quintiles),
                    "multipoles": [0, 1, 2],
                    "s": s,
                },
            )
            data = data.sel(
                quintiles=filters['quintiles'],
                multipoles=filters['multipoles'],
                s=slice(filters['s_min'], filters['s_max']),
            ).values.reshape(*data.shape[:1], -1)
            return np.cov(data, rowvar=False)

    @classmethod
    def get_covariance_test(
        cls,
        statistic,
        filters,
    ) -> np.array:
        if statistic == 'density_split':
            data = np.load(
                DATA_PATH
                / "ds_cross_xi_smu_zsplit_Rs20_c000.npy",
                allow_pickle=True,
            ).item()
            quintiles = range(5)
        else:
            raise ValueError(f'{statistic} is not implemented!')
        s = data["s"]
        data = data['multipoles']
        print(np.shape(data))
        if statistic == 'density_split':
            data = xr.DataArray(
                data, 
                dims=("phases", "quintiles", "multipoles", "s"), 
                coords={
                    "phases": list(range(len(data))),
                    "quintiles": list(quintiles),
                    "multipoles": [0, 1, 2],
                    "s": s,
                },
            )
            data = data.sel(
                quintiles=filters['quintiles'],
                multipoles=filters['multipoles'],
                s=slice(filters['s_min'], filters['s_max']),
            ).values.reshape(*data.shape[:1], -1)
            return np.cov(data, rowvar=False) / 64

    @classmethod
    def get_covariance_intrinsic(
        cls,
        statistic,
        filters,
    ) -> np.array:
        xi_test = []
        for cosmo in TEST_COSMOS:
            data = np.load(
                DATA_PATH
                / f'ds_cross_xi_smu_zsplit_Rs20_c{cosmo:03}_ph000.npy',
                allow_pickle=True
            ).item()
            s = data['s']
            data = data['multipoles']
            for hod in range(1, 999):
                xi_test.append(data[hod])
        xi_test = np.asarray(xi_test)
        xi_test = np.mean(xi_test, axis=1)
        xi_test = xr.DataArray(
            xi_test, 
            dims=("phases", "quintiles", "multipoles", "s"), 
            coords={
                "phases": list(range(len(xi_test))),
                "quintiles": range(5),
                "multipoles": [0, 1, 2],
                "s": s,
            },
        )
        xi_test = xi_test.sel(
            quintiles=filters['quintiles'],
            multipoles=filters['multipoles'],
            s=slice(filters['s_min'], filters['s_max']),
        ).values.reshape(*xi_test.shape[:1], -1)
        emulator = DensitySplit(quintiles=filters["quintiles"])
        xi_model = []
        for cosmo in TEST_COSMOS:
            data = np.genfromtxt(
                DATA_PATH
                / f'AbacusSummit_c{cosmo:03}_hod1000.csv',
                skip_header=1,
                delimiter=","
            )
            for hod in range(1, 999):
                params = torch.tensor(
                    data[hod],
                    dtype=torch.float32
                ).reshape(1, -1)
                xi = emulator.get_for_sample(
                    inputs=params,
                    filters=filters
                )
                xi_model.append(xi)
        xi_model = np.asarray(xi_model)
        return np.cov(xi_model - xi_test, rowvar=False)


def normalize_cov(cov):
    nbins = len(cov)
    corr = np.zeros_like(cov)
    for i in range(nbins):
        for j in range(nbins):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return corr


