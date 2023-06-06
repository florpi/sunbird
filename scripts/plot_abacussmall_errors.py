import numpy as np
from matplotlib.colors import colorConverter as cc
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sunbird.summaries import DensitySplitAuto, Bundle
import torch
from pathlib import Path
import sys
from sunbird.summaries import TPCF, DensitySplitCross, DensitySplitAuto
from sunbird.data.data_readers import Abacus, AbacusSmall
from sunbird.covariance import CovarianceMatrix, normalize_cov
from scipy.stats import median_abs_deviation, sigmaclip
from sunbird.inference import Inference
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
import json
import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='bossprior')
argparser.add_argument('--statistic', type=str, default='density_split_cross')
argparser.add_argument('--ell', type=int, default=0)
argparser.add_argument('--ds', type=int, default=4)

args = argparser.parse_args()

smin, smax = 1.0, 151

s = np.load('/global/homes/e/epaillas/code/sunbird/data/s.npy')
s = s[(s > smin) & (s < smax)]
statistic = 'density_split_cross'

slice_filters = {
    's': [smin, smax],
}
select_filters = {
    'quintiles': [args.ds],
    'multipoles': [args.ell],
}

data_reader = AbacusSmall(
    dataset='bossprior',
    slice_filters=slice_filters,
    select_filters=select_filters,
    statistics=[statistic],
)

summaries = data_reader.gather_summaries_for_covariance()

fig, ax = plt.subplots(figsize=(4, 3))
summaries_mean = np.mean(summaries, axis=0)
ax.plot(s, s**2 * summaries_mean, color='k', lw=2)

covariance = CovarianceMatrix(
    covariance_data_class='AbacusSmall',
    dataset='bossprior',
    statistics=[statistic],
    slice_filters=slice_filters,
    select_filters=select_filters,
)

cov_1 = covariance.get_covariance_data(volume_scaling=1)
err_1 = np.sqrt(np.diag(cov_1))

cov_8 = covariance.get_covariance_data(volume_scaling=8)
err_8 = np.sqrt(np.diag(cov_8))

cov_64 = covariance.get_covariance_data(volume_scaling=64)
err_64 = np.sqrt(np.diag(cov_64))

ax.fill_between(s, s**2 * (summaries_mean - err_1),
                s**2 * (summaries_mean + err_1), alpha=0.2, color='k', label='1')

ax.fill_between(s, s**2 * (summaries_mean - err_8),
                s**2 * (summaries_mean + err_8), alpha=0.2, color='royalblue', label='1/8')

ax.fill_between(s, s**2 * (summaries_mean - err_64),
                s**2 * (summaries_mean + err_64), alpha=0.2, color='crimson', label='1/64')

ax.legend()

ax.set_title(f'{args.statistic}, ell{args.ell}, Q{args.ds}')
ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
ax.set_ylabel(rf'$s^2 \xi_{args.ell}(s)\, [h^{{-2}}{{\rm Mpc^2}}]$')
plt.tight_layout()
plt.show()