"""
Figure 3: Multipoles
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib
from pathlib import Path
from sunbird.data.data_readers import CMASS
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import Bundle
from getdist import plots, MCSamples
import argparse

plt.style.use(['stylelib/science.mplstyle', 'stylelib/vibrant.mplstyle'])

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def read_dynesty_chain(filename):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights

def get_names_labels(param_space):
    names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
             'nrun', 'N_ur', 'w0_fld', 'wa_fld',
             'logM1', 'logM_cut', 'alpha', 'alpha_s',
             'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat']
    labels_dict = {
        "omega_b": r'\omega_{\rm b}', "omega_cdm": r'\omega_{\rm cdm}',
        "sigma8_m": r'\sigma_8', "n_s": r'n_s', "nrun": r'\alpha_s',
        "N_ur": r'N_{\rm ur}', "w0_fld": r'w_0', "wa_fld": r'w_a',
        "logM1": r'\log M_1', "logM_cut": r'\log M_{\rm cut}',
        "alpha": r'\alpha', "alpha_s": r'\alpha_{\rm vel, s}',
        "alpha_c": r'\alpha_{\rm vel, c}', "logsigma": r'\log \sigma',
        "kappa": r'\kappa', "B_cen": r'B_{\rm cen}', "B_sat": r'B_{\rm sat}',
    }
    if not 'w0wa' in param_space:
        names.remove('w0_fld')
        names.remove('wa_fld')
    if not 'nrun' in param_space:
        names.remove('nrun')
    if not 'Nur' in param_space:
        names.remove('N_ur')
    if 'noAB' in param_space:
        names.remove('B_cen')
        names.remove('B_sat')
    labels = [labels_dict[name] for name in names]
    return names, labels


args = argparse.ArgumentParser()
args.add_argument('--statistic', type=str, nargs="*", default=['density_split_cross'])
args.add_argument('--ell', type=int, nargs="*", default=[0])
args = args.parse_args()

for statistic in args.statistic:
    for ell in args.ell:
        print(statistic, ell)
        root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')
        chain_handle = f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn'

        names, labels = get_names_labels(chain_handle)
        chain_fn = root_dir / chain_handle / 'results.csv'
        data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
        chain = data[:, 4:]
        weights = np.exp(data[:, 1] - data[-1, 2])
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)

        # this reads the ML point from the chain
        parameters = {names[i]: samples[names[i]][-1] for i in range(len(names))}
        parameters['nrun'] = 0.0
        parameters['N_ur'] = 2.0328
        parameters['w0_fld'] = -1.0
        parameters['wa_fld'] = 0.0

        fig, ax = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        cmap = matplotlib.cm.get_cmap('coolwarm')
        colors = cmap(np.linspace(0.01, 0.99, 5))
        spow = 2

        quantiles = [0, 1, 3, 4] if 'density_split' in statistic else [5]

        for ds in quantiles:

            s = np.load('/pscratch/sd/e/epaillas/sunbird/data/s.npy')
            smin, smax = 0.7, 150
            s = s[(s > smin) & (s < smax)]

            slice_filters = {
                's': [smin, smax],
            }
            select_filters = {
                'quintiles': ds,
                'multipoles': ell,
            }

            datavector = CMASS(
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
                region='NGC'
            ).get_observation()

            cov = CovarianceMatrix(
                covariance_data_class='Patchy',
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
                path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/'
            )

            emulator = Bundle(
                summaries=[statistic],
                path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/',
            )

            model, error_model = emulator(
                param_dict=parameters,
                select_filters=select_filters,
                slice_filters=slice_filters,
            )

            cov_data = cov.get_covariance_data()
            cov_emu = cov.get_covariance_emulator()
            cov_sim = cov.get_covariance_simulation()
            cov_tot = cov_data + cov_emu + cov_sim
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
            ax[0].fill_between(s, s**spow*(model - error_model), s**spow*(model + error_model), color=color, alpha=0.4)

            ax[1].plot(s, (datavector - model)/error_tot, ls='-', color=color)

            dof = (len(datavector) - 13)
            chi2 = np.dot(datavector - model, np.linalg.inv(cov_tot)).dot(datavector - model)
            chi2_red = chi2 / dof
            print(f'{statistic} reduced chi2 = {chi2_red}')

        ax[0].axes.get_xaxis().set_visible(False)
        # ax[1].fill_between([-1, 160], -1, 1, color='grey', alpha=0.2)
        # ax[1].fill_between([-1, 160], -2, 2, color='grey', alpha=0.15)
        ax[1].set_ylim(-3, 3)
        ax[1].set_xlim(-0, 150)
        ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
        if statistic == 'tpcf':
            ax[0].set_ylabel(rf'$s^2\xi^{{\rm gg}}_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
            ax[1].set_ylabel(rf'$\Delta\xi^{{\rm gg}}_{ell}(s)/\sigma$')
        elif statistic == 'density_split_auto':
            ax[0].set_ylabel(rf'$s^2\xi^{{\rm qq}}_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
            ax[1].set_ylabel(rf'$\Delta\xi^{{\rm qq}}_{ell}(s)/\sigma$')
        elif statistic == 'density_split_cross':
            ax[0].set_ylabel(rf'$s^2\xi^{{\rm qg}}_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
            ax[1].set_ylabel(rf'$\Delta\xi^{{\rm qg}}_{ell}(s)/\sigma$')
        ax[1].set_ylabel(rf'$\Delta\xi(s)/\sigma$')
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
        plt.savefig(f'fig/pdf/multipoles_cmass_{statistic}_ell{ell}.pdf', dpi=300)