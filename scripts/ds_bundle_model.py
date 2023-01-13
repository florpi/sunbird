import numpy as np
from matplotlib.colors import colorConverter as cc
import matplotlib.pyplot as plt
from sunbird.summaries import DensitySplitAuto, Bundle
import torch
from pathlib import Path
import sys
from sunbird.covariance import CovarianceMatrix
import json
# plt.style.use(['enrique-science', 'bright'])


ells = [0]
ds = [3]

slice_filters = {
    's': [0.0, 151.],
}
select_filters = {
    'quintiles': ds,
    'multipoles': ells,
}

covariance = CovarianceMatrix(
    statistics=['density_split_cross'],
    slice_filters=slice_filters,
    select_filters=select_filters,
)
cov_data = covariance.get_covariance_data()
cov_data_frac = covariance.get_covariance_data(fractional=True)
err_data = np.sqrt(np.diag(cov_data))
err_data_frac = np.sqrt(np.diag(cov_data_frac))

cov_abs = covariance.get_covariance_emulator_error()
cov_frac = covariance.get_covariance_emulator_error(fractional=True)
err_abs = np.sqrt(np.diag(cov_abs))
err_frac = np.sqrt(np.diag(cov_frac))
err_frac_symmetrized = err_frac

with open(f"/global/homes/e/epaillas/code/sunbird/data/train_test_split.json", "r") as f:
    test_cosmologies = json.load(f)["test"]
xi_test = covariance.get_true_test(test_cosmologies=test_cosmologies)
inputs = covariance.get_inputs_test(test_cosmologies=test_cosmologies)
xi_model = covariance.get_emulator_predictions(inputs=inputs)


nmocks = [0]
fig, ax = plt.subplots(2, 1, figsize=(5, 6), height_ratios=[1, 1])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for nmock in nmocks:

    s = np.arange(0, len(xi_model[nmock]), 1)
    spow = 0

    if nmock == nmocks[0]:
        ax[0].plot(s, s**spow*xi_model[nmock], label='emulator prediction',
            color=colors[0])
        ax[0].errorbar(s, s**spow*xi_test[nmock], s**spow*err_data, label='data', ls='', 
            ms=5.0, marker='o', color=cc.to_rgba(colors[0], alpha=0.5), elinewidth=1.0, capsize=2.0,
            markeredgewidth=1, markeredgecolor=colors[0])
    else:
        ax[0].plot(s, s**spow*xi_model[nmock],
            color=colors[0])
        ax[0].errorbar(s, s**spow*xi_test[nmock], s**spow*err_data, ls='', 
            ms=5.0, marker='o', color=cc.to_rgba(colors[0], alpha=0.5), elinewidth=1.0, capsize=2.0,
            markeredgewidth=1, markeredgecolor=colors[0])

ax[1].fill_between(s, -err_data_frac, err_data_frac, color='dodgerblue', alpha=0.3, label='data error')
ax[1].fill_between(s, -err_frac_symmetrized, err_frac_symmetrized, color='plum', alpha=0.5, label='emulator error')
# ax[1].fill_between(s, 0, err_abs/err_data, color='plum', alpha=0.5, label='emulator error')
ax[1].set_ylim(-0.2, 0.2)

ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')

if spow == 2:
    ax[0].set_ylabel(r'$s^2 \xi_0(s) \,[h^{-2}{\rm Mpc^2}]$')
else:
    ax[0].set_ylabel(r'$\xi_0(s)$')

ax[1].set_ylabel('fractional error')
# ax[2].set_ylabel(r'$(\xi_{\rm emu} - \xi_{\rm test}) / \xi_{\rm test} / \sigma_{\rm data}$')
ax[0].legend(loc='best')
ax[1].legend(loc='lower left')
ax[0].axes.get_xaxis().set_visible(False)
# ax[1].set_ylim(0, 3)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
# plt.savefig(f'monopole_emulator_ds{ds+1}_abserr_s2.png', dpi=300)
plt.show()



