import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from cosmoprimo.fiducial import AbacusSummit

plt.style.use(['science.mplstyle', 'no-latex'])

redshift = 0.4

def read_dynesty_chain(filename,): 
    data = pd.read_csv(filename)
    # remove column
    data = data.drop(columns=['weights'])
    data = data.to_numpy()
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights


def get_true_params(cosmology, hod_idx):
    return dict(
        pd.read_csv(
            f"../../data/parameters/abacus/bossprior/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        ).iloc[hod_idx]
    )

args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/enrique/emulator_paper/')
args = args.parse_args()

chain_dir = Path(args.chain_dir)

chain_handles = [
    'abacus_cosmo0_hod26_tpcf_mae_vol64_smin0.70_smax150.00_m02_q0134',
    'abacus_cosmo0_hod26_density_split_cross_mae_vol64_smin0.70_smax150.00_m02_q0134',
    'abacus_cosmo0_hod26_density_split_cross_tpcf_mae_vol64_smin0.70_smax150.00_m02_q0134',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134_nosimerr',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134_noemuerr',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q04',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin30.00_smax150.00_m02_q0134',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax30.00_m02_q0134',
    'abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134',
] 

chain_labels = [
    'Galaxy 2PCF only',
    'DS CCF only',
    'DS CCF + Galaxy 2PCF',
    'no simulation error',
    'no emulator error',
    r'${\rm Q_0 + Q_4}$'' only',
    r'$s_{\rm max}=30\,h^{-1}{\rm Mpc}$',
    r'$s_{\rm min}=30\,h^{-1}{\rm Mpc}$',
    'baseline',
]
true_params = get_true_params(0,26)
true_params['fsigma8'] =  AbacusSummit(0).growth_rate(redshift)
print(true_params)

yvals = np.linspace(0, 10, len(chain_handles))
params_toplot = ['omega_cdm', 'sigma8_m', 'n_s', 'fsigma8']
labels_toplot = [r'$\omega_{\rm cdm}$', r'$\sigma_8$', r'$n_s$', r'$f\sigma_8$']

fig, ax = plt.subplots(1, len(params_toplot), figsize=(11, 4.5))
for iparam, param in enumerate(params_toplot):
    for ichain, chain_handle in enumerate(chain_handles):
        chain_fn = chain_dir / chain_handle / 'results.csv'
        if 'noabias' in chain_handle:
            names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
                     'nrun', 'N_ur', 'w_0', 'w_a',
                     'logM1', 'logM_cut', 'alpha', 'alpha_s',
                     'alpha_c', 'sigma', 'kappa',]
            labels = [r'\omega_b', r'\omega_{\rm cdm}', r'\sigma_8', 'n_s',
                      'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
                      r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa',]
        elif 'novelbias' in chain_handle:
            names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
                     'nrun', 'N_ur', 'w_0', 'w_a',
                     'logM1', 'logM_cut', 'alpha', 'sigma', 'kappa', 'B_cen', 'B_sat']
            labels = [r'\omega_b', r'\omega_{\rm cdm}', r'\sigma_8', 'n_s',
                      'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, c}',
                      r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}']
        else:
            names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
                     'nrun', 'N_ur', 'w_0', 'w_a',
                    'logM1', 'logM_cut', 'alpha', 'alpha_s',
                    'alpha_c', 'sigma', 'kappa', 'B_cen', 'B_sat']
            labels = [r'\omega_b', r'\omega_{\rm cdm}', r'\sigma_8', 'n_s',
                    'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
                    r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}']
        names.append('fsigma8')
        labels.append(r'f\sigma_8')


        chain, weights = read_dynesty_chain(chain_fn,)
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
        # print(samples.getTable(limit=1).tableTex())


        if ichain == len(chain_handles) - 1:
            ax[iparam].fill_betweenx(yvals, samples.mean(param) - samples.std(param),
                                     samples.mean(param) + samples.std(param), alpha=0.2)
            ax[iparam].plot([true_params[param]]*len(yvals), yvals, color='darkblue',alpha=0.3)

        ax[iparam].errorbar(samples.mean(param), yvals[ichain],
                       xerr=samples.std(param), marker='o',
                       ms=5.0, color='darkblue')

    ax[iparam].set_xlabel(labels_toplot[iparam], fontsize=20)
    ax[iparam].tick_params(axis='x', labelsize=13, rotation=45)
    if iparam > 0:
        ax[iparam].axes.get_yaxis().set_visible(False)
    else:
        ax[iparam].set_yticks(yvals)
        ax[iparam].set_yticklabels(chain_labels, minor=False, rotation=0, fontsize=13)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig('figures/pdf/whisker.pdf', bbox_inches='tight')
plt.savefig(f"figures/png/whisker.png", bbox_inches='tight', dpi=300)