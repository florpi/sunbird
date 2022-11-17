from sunbird.summaries import DensitySplit, TPCF
import numpy as np
from pathlib import Path
import xarray as xr
import torch
from matplotlib import pyplot as plt


DATA_PATH = Path('/home/epaillas/data/ds_boss/Patchy/ds_cross_xi_smu/compressed/z0.46-0.6/NGC/lin_binning/')
TEST_SMALL_PATH = Path('/home/epaillas/data/ds_boss/HOD/ds_cross_xi_smu/compressed/small/z0.5/linear_binning')
TEST_BASE_PATH = Path('/home/epaillas/data/ds_boss/HOD/ds_cross_xi_smu/compressed/z0.5/full_ap/linear_binning')
COSMO_PATH = Path('/home/epaillas/data/ds_boss/HOD/cosmologies/')


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
                TEST_SMALL_PATH
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
                TEST_BASE_PATH
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
                COSMO_PATH
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


if __name__ == '__main__':
    filters = {
        's_min': 0,
        's_max': 150,
        'multipoles': [0, 1],
        'quintiles': [0, 1, 3, 4],
    }

    TEST_COSMOS = list(range(5))

    fig, ax = plt.subplots()
    cov_data = CovarianceMatrix.get_covariance_data(
        statistic='density_split', filters=filters)
    ax.imshow(normalize_cov(cov_data), origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xlabel("bin number")
    ax.set_ylabel("bin number")
    ax.set_title('data error')
    plt.tight_layout()
    plt.savefig('covariance_data.png', dpi=300)

    fig, ax = plt.subplots()
    cov_test = CovarianceMatrix.get_covariance_test(
        statistic='density_split', filters=filters)
    ax.imshow(normalize_cov(cov_test), origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xlabel("bin number")
    ax.set_ylabel("bin number")
    ax.set_title('test set error')
    plt.tight_layout()
    plt.savefig('covariance_test.png', dpi=300)

    fig, ax = plt.subplots(figsize=(4, 4))
    cov_intrinsic = CovarianceMatrix.get_covariance_intrinsic(
        statistic='density_split', filters=filters)
    ax.imshow(normalize_cov(cov_intrinsic), origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xlabel("bin number")
    ax.set_ylabel("bin number")
    ax.set_title('intrinsic emulator error')
    plt.tight_layout()
    plt.savefig('covariance_intrinsic.png', dpi=300)

