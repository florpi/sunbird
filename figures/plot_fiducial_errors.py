import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import colorConverter as cc
import matplotlib.pyplot as plt
from sunbird.summaries import DensitySplitAuto, DensitySplitCross
import torch
from sunbird.covariance import CovarianceMatrix
plt.style.use(['science', 'bright', 'no-latex'])

model = 'same_hods'
corr_type = 'auto'
DATA_PATH = Path(__file__).parent.parent / "data/"
# Get errors

slice_filters = {
    's': [0., 150.],
}
select_filters = {
    'multipoles': [0, 1],
    'quintiles': [0,1,3,4],
}
covariance = CovarianceMatrix(
    statistics=[f'density_split_{corr_type}'], 
    slice_filters=slice_filters,
    select_filters=select_filters,
)

cov_data = covariance.get_covariance_data()
cov_test = covariance.get_covariance_emulator_error()
std_data = np.sqrt(np.diag(cov_data))
std_data = std_data.reshape((4,2,-1))
std_test = np.sqrt(np.diag(cov_test))
std_test = std_test.reshape((4,2,-1))

spow = 2
if corr_type == 'auto':
    emu = DensitySplitAuto()
else:
    emu = DensitySplitCross()

test_idx = [0,1,2,3,4,13,]
xi_datas, xi_emus = [], []
for idx in test_idx:
    # read a given cosmo and hod
    data = np.load(
        DATA_PATH
        / f"clustering/different_hods/ds/gaussian/ds_{corr_type}_xi_smu_zsplit_Rs20_c{str(idx).zfill(3)}_ph000.npy",
        allow_pickle=True,
    ).item()
    quintiles = range(5)
    s = data["s"]
    multipoles = range(3)
    xi_data = data['multipoles']
    xi_data = np.mean(xi_data,axis=1)
    xi_data = xi_data[:,[0,1,3,4]]
    xi_data = xi_data[:, :, [0,1], :]
    # get emulator prediction
    params = pd.read_csv(
            DATA_PATH
            / f"parameters/different_hods/AbacusSummit_c{str(idx).zfill(3)}_hod1000.csv"
    ).to_numpy()
    params = torch.tensor(params, dtype=torch.float32)
    xi_emu = emu.get_for_batch_inputs(
        params,
        None,
        None,
    ).reshape((1000, 4, 2,-1))

    xi_datas.append(xi_data)
    xi_emus.append(xi_emu)

xi_datas = np.array(xi_datas)
xi_emus = np.array(xi_emus)
xi_datas = xi_datas.reshape((-1, xi_datas.shape[2], xi_datas.shape[3], xi_datas.shape[4]))
xi_emus = xi_emus.reshape((-1, xi_emus.shape[2], xi_emus.shape[3], xi_emus.shape[4]))

std_intrinsic = np.std(xi_emus - xi_datas, axis=0) / std_data
titles = [
    r'$\mathrm{DS}_1$',
    r'$\mathrm{DS}_2$',
    r'$\mathrm{DS}_4$',
    r'$\mathrm{DS}_5$',
]
n_to_plot = 10
idx_to_show = np.random.randint(low=0, high=len(xi_datas), size=n_to_plot)
for ds in [0,1,2,3]:
    for multipole in [0,1]:
        fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True)#height_ratios=[1, 1])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, idx in enumerate(idx_to_show):
            c = ax[0].plot(s, 
                    s**spow*xi_emus[idx,ds, multipole], 
                    label='emulator prediction' if i == 0 else None,
                )#color=colors[i], )
            ax[0].errorbar(s, s**spow*xi_datas[idx, ds, multipole], s**spow*std_data[ds, multipole], 
                label='data' if i == 0 else None, ls='', 
                ms=1.0, marker='o', color=cc.to_rgba(c[0].get_color(), alpha=0.5), elinewidth=1.0, capsize=2.0,
                markeredgewidth=1, markeredgecolor=c[0].get_color())

        ax[1].fill_between(s, 0, (std_test/std_data)[ds,multipole], color='grey', alpha=0.5, label='training sample error')
        ax[1].fill_between(s, 0, (std_intrinsic)[ds,multipole], color='plum', alpha=0.5, label='emulator error')
        ax[1].hlines(1.0, 0, 150, ls='--', color='grey')
        ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
        if spow == 2:
            ax[0].set_ylabel(r'$s^2 \xi_0(s) \,[h^{-2}{\rm Mpc^2}]$')
        else:
            ax[0].set_ylabel(r'$\xi_0(s)$')
        ax[1].set_ylabel(r'$(\xi_{\rm emu} - \xi_{\rm data}) / \sigma_{\xi_{\rm data}}$')
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[1].set_ylim(0, 10)
        fig.suptitle(titles[ds])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(f'plots/errors/{corr_type}_{model}_m{multipole}_error_ds{ds+1}.png', dpi=300)
