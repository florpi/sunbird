import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sunbird.data.data_readers import Abacus 
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import Bundle
import scienceplots
import argparse
from utils import lighten_color, get_names_labels

plt.style.use(['science', 'vibrant'])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--statistic', type=str, nargs="*", default=['density_split_cross', 'density_split_auto', 'tpcf'])
    args.add_argument('--ell', type=int, nargs="*", default=[0,2,])
    args.add_argument('--loss', type=str, default='learned_gaussian')
    args = args.parse_args()

    for statistic in args.statistic:
        for ell in args.ell:
            print(statistic, ell)
            fig, ax = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
            cmap = matplotlib.cm.get_cmap('coolwarm')
            colors = cmap(np.linspace(0.01, 0.99, 5))
            spow = 2
            quantiles = [0, 1, 3, 4] if 'density_split' in statistic else [5]
            cosmology, hod_idx = 0, 26

            for ds in quantiles:
                slice_filters = {'s': [0.7,150.]}
                select_filters = {
                    'quintiles': ds,
                    'multipoles': ell,
                }
                abacus = Abacus(
                    statistics=[statistic],
                    slice_filters=slice_filters,
                    select_filters=select_filters,
                )
                datavector = abacus.get_observation(cosmology=cosmology, hod_idx= hod_idx,) 
                parameters = abacus.get_all_parameters(
                    cosmology=cosmology,
                ).iloc[hod_idx]

                emulator = Bundle(
                    summaries=[statistic],
                    loss=args.loss,
                )
                model, variance_model = emulator(
                    param_dict=parameters,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                )
                s = emulator.coordinates['s']
                cov = CovarianceMatrix(
                    covariance_data_class='AbacusSmall',
                    statistics=[statistic],
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                    emulators = emulator.all_summaries,

                )
                cov_data = cov.get_covariance_data(volume_scaling=64.)
                cov_emu = cov.get_covariance_emulator()
                cov_sim = cov.get_covariance_simulation()
                error_data = np.sqrt(np.diag(cov_data))
                error_emu = np.sqrt(np.diag(cov_emu))
                error_sim = np.sqrt(np.diag(cov_sim))
                error_model = np.sqrt(error_sim**2 + error_emu**2)
                error_tot = np.sqrt(error_data**2 + error_emu**2 + error_sim**2)

                color = colors[ds] if 'density_split' in statistic else '#404040'

                ax[0].errorbar(s, s**spow*datavector, s**spow*error_data, marker='o',
                            ms=3.0, ls='', color=color, elinewidth=1.0,
                            capsize=0.0, markeredgewidth=1.0,
                            markerfacecolor=lighten_color(color, 0.5),
                            markeredgecolor=color,)

                ax[0].plot(s, s**spow*model, ls='-', color=color)
                ax[0].fill_between(s, s**spow*(model - variance_model), s**spow*(model + variance_model), color=color, alpha=0.2)
                ax[0].fill_between(s, s**spow*(model - variance_model), s**spow*(model + variance_model), color=color, alpha=0.2)

                ax[1].plot(s, (datavector - model)/error_data, ls='-', color=color)

            ax[0].axes.get_xaxis().set_visible(False)
            ax[1].fill_between([-1, 160], -1, 1, color='grey', alpha=0.2)
            ax[1].fill_between([-1, 160], -2, 2, color='grey', alpha=0.15)
            ax[1].set_ylim(-4, 4)
            ax[1].set_xlim(-0, 150)
            ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
            ax[0].set_ylabel(rf'$s^2\xi_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
            ax[1].set_ylabel(rf'$\Delta\xi_{ell}(s)/\sigma$')
            if ell == 0:
                multipole = 'monopole'
            elif ell == 2:
                multipole = 'quadrupole'
            if statistic == 'tpcf':
                title = f'Galaxy 2PCF {multipole}'
            elif statistic == 'density_split_auto':
                title = f'Density-split ACF {multipole}'
            elif statistic == 'density_split_cross':
                title = f'Density-split CCF {multipole}'
            ax[0].set_title(title, fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            plt.savefig(f'figures/png/F1_multipoles_c0_{statistic}_ell{ell}.png', dpi=300)
            plt.savefig(f'figures/pdf/F1_multipoles_c0_{statistic}_ell{ell}.pdf', dpi=300)
            plt.show()