import numpy as np
from matplotlib.colors import colorConverter as cc
import matplotlib.pyplot as plt
from sunbird.summaries import DensitySplitAuto, Bundle
from pathlib import Path
from sunbird.summaries import TPCF, DensitySplitCross, DensitySplitAuto
from sunbird.data.data_readers import Abacus
from sunbird.covariance import CovarianceMatrix, normalize_cov
from scipy.stats import median_abs_deviation, sigmaclip
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
argparser.add_argument('--path_to_models', action='store', nargs='*', type=str,
                       default=['/pscratch/sd/e/epaillas/sunbird/trained_models/best'])
argparser.add_argument('--volume_scaling', type=int, default=8)
argparser.add_argument('--output_fn', type=str, default=None)
args = argparser.parse_args()

smin, smax = 1.0, 151

s = np.load('/pscratch/sd/e/epaillas/sunbird/data/s.npy')
s = s[(s > smin) & (s < smax)]


slice_filters = {
    's': [smin, smax],
}
select_filters = {
    'quintiles': [args.ds],
    'multipoles': [args.ell],
}

fig, ax = plt.subplots()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, path_to_models in enumerate(args.path_to_models):

    covariance = CovarianceMatrix(
        covariance_data_class='AbacusSmall',
        dataset='bossprior',
        statistics=[args.statistic],
        slice_filters=slice_filters,
        select_filters=select_filters,
        path_to_models=Path(path_to_models),
    )

    cov_data = covariance.get_covariance_data(volume_scaling=args.volume_scaling)
    err_data = np.sqrt(np.diag(cov_data))

    cov_emu = covariance.get_covariance_emulator()
    err_emu = np.sqrt(np.diag(cov_emu))

    label = f'{path_to_models.split("/")[-2]}'
    ax.fill_between(s, 0, err_emu/err_data, alpha=0.5, label=label)
        
ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax.set_ylabel(r'$(\xi_{\rm emu} - \xi_{\rm test})/ \sigma_{\rm data}$', fontsize=15)
ax.legend(loc='best')
ax.set_title(f'{args.statistic} Q{args.ds}', fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.tight_layout()
if args.output_fn is not None:
    plt.savefig(args.output_fn, dpi=300)
plt.show()
