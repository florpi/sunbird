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
argparser.add_argument('--path_to_models', type=str, default='/pscratch/sd/e/epaillas/sunbird/trained_models/best')
argparser.add_argument('--volume_scaling', type=int, default=8)
args = argparser.parse_args()

smin, smax = 1.0, 151

s = np.load('/pscratch/sd/e/epaillas/sunbird/data/s.npy')
s = s[(s > smin) & (s < smax)]
statistic = 'density_split_cross'

slice_filters = {
    's': [smin, smax],
}
select_filters = {
    'quintiles': [args.ds],
    'multipoles': [args.ell],
}

covariance = CovarianceMatrix(
    covariance_data_class='AbacusSmall',
    dataset='bossprior',
    statistics=[statistic],
    slice_filters=slice_filters,
    select_filters=select_filters,
    path_to_models=Path(args.path_to_models),
)

cov_data = covariance.get_covariance_data(volume_scaling=args.volume_scaling)
err_data = np.sqrt(np.diag(cov_data))

cov_emu = covariance.get_covariance_emulator()
err_emu = np.sqrt(np.diag(cov_emu))

with open(f"/global/homes/e/epaillas/code/sunbird/data/train_test_split.json", "r") as f:
    test_cosmologies = json.load(f)["test"]
xi_test = covariance.get_true_test(test_cosmologies=test_cosmologies)
inputs = covariance.get_inputs_test(test_cosmologies=test_cosmologies)


xi_model = covariance.get_emulator_predictions(inputs=inputs)


fig, ax = plt.subplots(2, 1, figsize=(5, 6), height_ratios=[1, 1])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

spow = 2
nmock = 26

model = xi_model[nmock]

ax[0].plot(s, s**spow*model, label='emulator prediction',
    color=colors[0])
ax[0].errorbar(s, s**spow*xi_test[nmock], s**spow*err_data, label='test simulation', ls='',
    ms=5.0, marker='o', color=cc.to_rgba(colors[0], alpha=0.5), elinewidth=1.0, capsize=2.0,
    markeredgewidth=1, markeredgecolor=colors[0])
        
ax[1].fill_between(s, 0, err_emu/err_data, color='plum', alpha=0.5, label='68% C.I.')
        
    
ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)

if spow == 2:
    ax[0].set_ylabel(rf'$s^2 \xi_{args.ell}(s) \,[h^{{-2}}{{\rm Mpc^2}}]$', fontsize=15)
else:
    ax[0].set_ylabel(rf'$\xi_{args.ell}(s)$', fontsize=15)

ax[1].set_ylabel(r'$(\xi_{\rm emu} - \xi_{\rm test})/ \sigma_{\rm data}$', fontsize=15)
ax[0].legend(loc='best')
ax[1].legend(loc='upper right')
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_title(f'{statistic} Q{args.ds}', fontsize=15)
for aa in ax:
    aa.tick_params(axis='x', labelsize=15)
    aa.tick_params(axis='y', labelsize=15)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
#plt.savefig(f'monopole_emulator_ds{ds}.png', dpi=300)
plt.show()
