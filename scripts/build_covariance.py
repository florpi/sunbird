from sunbird.summaries import DensitySplit, TPCF
import numpy as np
from pathlib import Path
import xarray as xr
import torch
from matplotlib import pyplot as plt
from sunbird.covariance import CovarianceMatrix


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

