"""
Figure 11: Whisker plot for CMASS
"""
import matplotlib.pyplot as plt
import numpy as np
import corner
from getdist import plots, MCSamples
import getdist
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sunbird.cosmology.growth_rate import Growth
import pandas as pd
from cosmoprimo.fiducial import AbacusSummit
getdist.chains.print_load_details = False

plt.style.use(['stylelib/science.mplstyle'])

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

def read_dynesty_chain(filename, add_fsigma8=True, redshift=0.5):
    data = pd.read_csv(filename)
    if add_fsigma8:
        data['fsigma8'] = growth.get_fsigma8(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
            z=redshift,
        )
    data = data.to_numpy()
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights

def get_true_params(cosmology, hod_idx):
    return dict(
        pd.read_csv(
            f"/pscratch/sd/e/epaillas/sunbird/data/parameters/abacus/bossprior/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        ).iloc[hod_idx]
    )

def get_names_labels(param_space, add_fsigma8=True):
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
        "fsigma8": r'f\sigma_8',
    }
    if 'base' in param_space:
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
        if 'noVB' in param_space:
            names.remove('alpha_c')
            names.remove('alpha_s')
    if add_fsigma8:
        names.append('fsigma8')
    labels = [labels_dict[name] for name in names]
    return names, labels


redshift = 0.5
growth = Growth(emulate=True,)
true_params = get_true_params(0,26)
pred_growth = growth.get_growth(
    omega_b = np.array(true_params['omega_b']).reshape(1,1),
    omega_cdm = np.array(true_params['omega_cdm']).reshape(1,1),
    sigma8 = np.array(true_params['sigma8_m']).reshape(1,1),
    n_s = np.array(true_params['n_s']).reshape(1,1),
    N_ur = np.array(true_params['N_ur']).reshape(1,1),
    w0_fld = np.array(true_params['w0_fld']).reshape(1,1),
    wa_fld = np.array(true_params['wa_fld']).reshape(1,1),
    z=redshift,
)
cosmo = AbacusSummit(0)
true_params['fsigma8'] = cosmo.sigma8_z(redshift) * cosmo.growth_rate(redshift)

# Load the chain
root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/enrique')

chain_handles = [
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn_noemuerr_nosimerr',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn_noVB',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn_noAB',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_full_bbn',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_Nur_bbn',
    f'cmass_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn',
    f'cmass_density_split_cross_density_split_auto_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn',
    f'cmass_density_split_cross_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q04_base_bbn',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m0_q0134_base_bbn',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base',
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_bbn',
] 

chain_labels = [
    'no model error',
    'no velocity bias',
    'no assembly bias',
    r'$w\Lambda{\rm CDM} + \{ \alpha_s, N_{\rm ur}\}$',
    'base 'r'$\Lambda{\rm CDM} + N_{\rm ur}$',
    'galaxy 2PCF',
    'DS CCF + DS ACF',
    'DS CCF + galaxy 2PCF',
    r'${\rm Q_0}\, \& \, {\rm Q_4}$'' only',
    'monopole only',
    'uniform 'r'$\omega_{\rm b}$'' prior',
    'baseline',
]

color = '#AA3377'
yvals = np.linspace(0, 10, len(chain_handles))
params_toplot = ['fsigma8', 'omega_cdm', 'sigma8_m', 'n_s']
labels_toplot = [r'$f\sigma_8$', r'$\omega_{\rm cdm}$', r'$\sigma_8$', r'$n_s$']

fig, ax = plt.subplots(1, len(params_toplot), figsize=(8, 6))
for iparam, param in enumerate(params_toplot):
    for ichain, chain_handle in enumerate(chain_handles):
        chain_dir = root_dir / chain_handle
        chain_fn = chain_dir / 'results.csv'
        names, labels = get_names_labels(chain_handle)

        chain, weights = read_dynesty_chain(chain_fn)
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
        # print(samples.getTable(limit=1).tableTex())

        ax[iparam].errorbar(samples.mean(param), yvals[ichain],
                       xerr=samples.std(param), marker='o', capsize=3, color=color,
                       ms=5.0, markerfacecolor=lighten_color(color), markeredgecolor=color,
                       elinewidth=1.5)

        if ichain == len(chain_handles) - 1:
            ax[iparam].fill_betweenx(yvals, samples.mean(param) - samples.std(param),
                                     samples.mean(param) + samples.std(param),
                                     color='grey', alpha=0.1)

    ax[iparam].set_xlabel(labels_toplot[iparam], fontsize=20)
    ax[iparam].tick_params(axis='x', labelsize=13, rotation=45)
    if iparam > 0:
        ax[iparam].axes.get_yaxis().set_visible(False)
    else:
        ax[iparam].set_yticks(yvals)
        ax[iparam].set_yticklabels(chain_labels, minor=False, rotation=0, fontsize=13)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig('fig/pdf/whisker_cmass.pdf', bbox_inches='tight')