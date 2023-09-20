"""
Figure 9: Scale cuts
"""
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
from pathlib import Path
from sunbird.cosmology.growth_rate import Growth
import pandas as pd
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


truth =  {
    "omega_b": 0.02303,
    "omega_cdm": 0.1171,
    "sigma8_m": 0.82,
    "n_s": 0.96,
    "nrun": 0.0,
    "N_ur": 2.046,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "h": 0.7,
}

redshift = 0.5
growth = Growth(emulate=True,)
pred_fsigma8 = growth.get_fsigma8(
    omega_b = np.array(truth['omega_b']).reshape(1,1),
    omega_cdm = np.array(truth['omega_cdm']).reshape(1,1),
    sigma8 = np.array(truth['sigma8_m']).reshape(1,1),
    n_s = np.array(truth['n_s']).reshape(1,1),
    N_ur = np.array(truth['N_ur']).reshape(1,1),
    w0_fld = np.array(truth['w0_fld']).reshape(1,1),
    wa_fld = np.array(truth['wa_fld']).reshape(1,1),
    z=redshift,
)
print(pred_fsigma8)
truth['fsigma8'] = pred_fsigma8[0]

# Load the chain
root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/enrique')

scales = [60, 50, 40, 30, 20, 10, 5.0, 0.7][::-1]

chain_handles_nseries = [
    f'nseries_cutsky_ph0_density_split_cross_density_split_auto_tpcf_mae_patchycov_vol1_smin{scales:.2f}_smax150.00_m02_q0134_base_bbn' for scales in scales
] 
chain_handles_cmass_tpcf = [
    f'cmass_tpcf_mae_patchycov_smin{scales:.2f}_smax150.00_m02_q0134_base_bbn_percival' for scales in scales
] 

chain_handles_cmass_baseline = [
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin{scales:.2f}_smax150.00_m02_q0134_base_bbn_percival3' for scales in scales
] 

chain_labels = scales

color = 'firebrick'
xvals = np.linspace(0, 10, len(chain_handles_nseries))
params_toplot = ['omega_cdm', 'sigma8_m', 'n_s', 'fsigma8']
labels_toplot = [r'$\omega_{\rm cdm}$', r'$\sigma_8$', r'$n_s$', r'$f\sigma_8$']

fig, ax = plt.subplots(len(params_toplot), 1, figsize=(6, 6.5))
for iparam, param in enumerate(params_toplot):
    drift_low = []
    drift_high = []
    for ichain, chain_handle in enumerate(chain_handles_nseries):
        chain_dir = root_dir / chain_handle
        chain_fn = chain_dir / 'results.csv'
        names, labels = get_names_labels(chain_handle)

        chain, weights = read_dynesty_chain(chain_fn)
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
        # print(samples.getTable(limit=1).tableTex())

        ax[iparam].errorbar(xvals[ichain], samples.mean(param),
                            yerr=samples.std(param), marker='o',
                            ms=5.0, color=color, elinewidth=1.5, markeredgecolor=color,
                            markeredgewidth=1.0, markerfacecolor=lighten_color(color, 0.5))
        
        drift_low.append(samples.mean(param) - samples.std(param))
        drift_high.append(samples.mean(param) + samples.std(param))
                            
    ax[iparam].fill_between(xvals, drift_low, drift_high,
                                color=color, alpha=0.1)

    if param in truth:
        ax[iparam].axhline(truth[param], color='k', ls='--', lw=1.0)

    ax[iparam].set_xlabel(r'$s_{\rm min}$ [Mpc/h]', fontsize=20)
    ax[iparam].set_ylabel(labels_toplot[iparam], fontsize=20)
    if iparam < 3:
        ax[iparam].axes.get_xaxis().set_visible(False)
    else:
        ax[iparam].set_xticks(xvals)
        ax[iparam].set_xticklabels(chain_labels, minor=False, rotation=0, fontsize=15)


color = 'dodgerblue'
for iparam, param in enumerate(params_toplot):
    drift_low = []
    drift_high = []
    for ichain, chain_handle in enumerate(chain_handles_cmass_tpcf):
        chain_dir = root_dir / chain_handle
        chain_fn = chain_dir / 'results.csv'
        names, labels = get_names_labels(chain_handle)

        chain, weights = read_dynesty_chain(chain_fn)
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)

        ax[iparam].errorbar(xvals[ichain]+0.2, samples.mean(param),
                       yerr=samples.std(param), marker='s',
                       ms=5.0, color='dodgerblue', elinewidth=1.5,
                       markeredgecolor='dodgerblue', markeredgewidth=1.0,
                       markerfacecolor=lighten_color('dodgerblue', 0.5))
                
        drift_low.append(samples.mean(param) - samples.std(param))
        drift_high.append(samples.mean(param) + samples.std(param))
                            
    ax[iparam].fill_between(xvals + 0.2, drift_low, drift_high,
                                color=color, alpha=0.1)

color = 'forestgreen'
for iparam, param in enumerate(params_toplot):
    drift_low = []
    drift_high = []
    for ichain, chain_handle in enumerate(chain_handles_cmass_baseline):
        chain_dir = root_dir / chain_handle
        chain_fn = chain_dir / 'results.csv'
        names, labels = get_names_labels(chain_handle)

        chain, weights = read_dynesty_chain(chain_fn)
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
        # print(samples.getTable(limit=1).tableTex())

        ax[iparam].errorbar(xvals[ichain]+0.2, samples.mean(param),
                       yerr=samples.std(param), marker='s',
                       ms=5.0, color=color, elinewidth=1.5,
                       markeredgecolor=color, markeredgewidth=1.0,
                       markerfacecolor=lighten_color(color, 0.5))
                
        drift_low.append(samples.mean(param) - samples.std(param))
        drift_high.append(samples.mean(param) + samples.std(param))
                            
    ax[iparam].fill_between(xvals + 0.2, drift_low, drift_high,
                                color=color, alpha=0.1)

ax[0].errorbar(np.nan, np.nan, yerr=np.nan, marker='o', ls='', ms=5.0,
               color='firebrick', elinewidth=1.5, label='Nseries',
               markeredgecolor='firebrick', markeredgewidth=1.0,
               markerfacecolor=lighten_color('firebrick', 0.5))
ax[0].errorbar(np.nan, np.nan, yerr=np.nan, marker='s', ls='', ms=5.0,
               color='dodgerblue', elinewidth=1.5, label='CMASS 2PCF',
               markeredgecolor='dodgerblue', markeredgewidth=1.0,
               markerfacecolor=lighten_color('dodgerblue', 0.5))

ax[0].errorbar(np.nan, np.nan, yerr=np.nan, marker='s', ls='', ms=5.0,
               color='forestgreen', elinewidth=1.5, label='CMASS baseline',
               markeredgecolor='forestgreen', markeredgewidth=1.0,
               markerfacecolor=lighten_color('forestgreen', 0.5))

ax[0].legend(bbox_to_anchor=(0.5, 1.5), loc='upper center',
        frameon=False, fontsize=18, ncols=4, handletextpad=0.0,
        columnspacing=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig('fig/pdf/scalecuts.pdf', bbox_inches='tight')
plt.show()