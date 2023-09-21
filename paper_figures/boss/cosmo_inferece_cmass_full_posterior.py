from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sunbird.cosmology.growth_rate import Growth
import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def read_dynesty_chain(filename, add_fsigma8=True, add_Omega_m=True, redshift=0.5):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
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
    if add_Omega_m:
        data['H0'] = growth.get_emulated_h(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
        ) * 100
        data['Omega_m'] = growth.Omega_m0(
            data['omega_cdm'],
            data['omega_b'],
            data['H0'] / 100,
        )
    data = data.to_numpy()
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    loglikes = data[:, 0] * -1
    return chain, weights, loglikes

def get_names_labels(param_space, add_fsigma8=True, add_Omega_m=True):
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
        "fsigma8": r'f\sigma_8', "Omega_m": r'\Omega_{\rm m}', "H0": r'H_0',
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
    if add_fsigma8:
        names.append('fsigma8')
    if add_Omega_m:
        names.append('H0')
        names.append('Omega_m')
    labels = [labels_dict[name] for name in names]
    return names, labels


args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')
args.add_argument('--param_space', type=str, default='cosmo')
args = args.parse_args()

chain_dir = Path(args.chain_dir)

smin = 0.7
smax = 150
redshift = 0.525
growth = Growth(emulate=True,)
param_space = 'base_bbn'
names, labels = get_names_labels(param_space)

chain_handles = [
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin{smin:.2f}_smax{smax:.2f}_m02_q0134_{param_space}',
]
chain_labels = [
    'density-split + galaxy 2PCF',
]

priors = {
    "omega_b": [0.0207, 0.0243],
    "omega_cdm": [0.1032, 0.140],
    "sigma8_m": [0.678, 0.938],
    "n_s": [0.9012, 1.025],
    "nrun": [-0.038, 0.038],
    "N_ur": [1.188, 2.889],
    "w0_fld": [-1.22, -0.726],
    "wa_fld": [-0.628, 0.621],
    "logM1": [13.2, 14.4],
    "logM_cut": [12.4, 13.3],
    "alpha": [0.7, 1.5],
    "alpha_s": [0.7, 1.3],
    "alpha_c": [0.0, 0.5],
    "logsigma": [-3.0, 0.0],
    "kappa": [0.0, 1.5],
    "B_cen": [-0.5, 0.5],
    "B_sat": [-1.0, 1.0],
}


params_toplot = [
    'omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
    'Omega_m', 'H0',
    'logM_cut', 'logM1', 'alpha',
    'logsigma', 'kappa',
    'alpha_s', 'alpha_c',
    'B_cen', 'B_sat',
]

samples_list = []

for i in range(len(chain_handles)):
    chain_fn = chain_dir / chain_handles[i] / 'results.csv'
    chain, weights, loglikes = read_dynesty_chain(chain_fn)
    samples = MCSamples(samples=chain, weights=weights, labels=labels,
                        names=names, ranges=priors, loglikes=loglikes,)
    samples_list.append(samples)
    print(samples.getLikeStats())
    print('Maximum likelihood:')
    print([f'{name} {samples[name][-1]:.4f}' for name in names])
    print('Standard deviation / mean:')
    print([f'{name} {samples.std(name) / samples.mean(name) * 100:.1f}' for name in names])
    print(samples.getTable(limit=1).tableTex())


param_limits = {
    'omega_cdm': [0.112, 0.1293],
    # 'sigma8_m': [0.68, 0.92],
    # 'n_s': [0.9012, 1.0],
    'logM_cut': [12.4, 12.9,],
    'logM1': [13.2, 14.0],
    'logsigma': [-0.8, -0.3],
    'B_cen': [-0.5, -0.1],
    'B_sat': [-1.0, 0.5],
    'kappa': [0.0, 1.3],
    'alpha_s': [0.7, 1.0],
}

colors = ['lightcoral', 'royalblue', 'orange']
bright = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
retro = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']

g = plots.get_subplot_plotter(width_inch=8)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth = 2.0
g.settings.linewidth_contour = 3.0
g.settings.legend_fontsize = 15
g.settings.axes_fontsize = 13
g.settings.axes_labelsize = 15
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = ['dimgrey']

g.triangle_plot(
    roots=samples_list,
    params=params_toplot,
    filled=True,
    param_limits=param_limits
)
output_fn = f'fig/pdf/{args.param_space}_inference_cmass_full.pdf'
plt.savefig(output_fn, bbox_inches='tight')
