"""
Figure 6: extended-LCDM (right panel)
"""
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.style.use(['stylelib/science.mplstyle', 'stylelib/retro.mplstyle'])
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [palette[0], 'dimgrey', palette[1]]

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
    if not 'w0' in param_space:
        names.remove('w0_fld')
    if not 'wa' in param_space:
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


args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='/pscratch/sd/e/epaillas/sunbird/chains/enrique')
args = args.parse_args()

priors = {
    "omega_b": [0.0207, 0.0243],
    "omega_cdm": [0.1032, 0.140],
    "sigma8_m": [0.678, 0.938],
    "n_s": [0.9012, 1.025],
    "nrun": [-0.038, 0.038],
    "N_ur": [1.188, 2.889],
    "w0_fld": [-1.22, -0.726],
    "wa_fld": [-0.628, 0.621]
}

samples_list = []
samples_planck = loadMCSamples('/pscratch/sd/e/epaillas/planck/base_w/plikHM_TTTEEE_lowl_lowE/base_w_plikHM_TTTEEE_lowl_lowE',
                                    settings={'ignore_rows': 0.3})
samples_planck.addDerived(samples_planck.getParams().w, name='w0_fld', label='w_0')
samples_list.append(samples_planck)

param_space = f'base_w0'
names, labels = get_names_labels(param_space)

chain_dir = Path(args.chain_dir)
chain_handle = f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_{param_space}_bbn'
chain_fn = chain_dir / chain_handle / 'results.csv'
data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
chain = data[:, 4:]
loglikes = data[:, 0] * -1
weights = np.exp(data[:, 1] - data[-1, 2])
samples = MCSamples(samples=chain, weights=weights, labels=labels,
                    names=names, ranges=priors, loglikes=loglikes,)
samples_list.append(samples)
print(samples.getTable(limit=1).tableTex())
# print(samples.getLikeStats())
print([f'{name} {samples[name][-1]:.4f}' for name in names])

samples_planck = loadMCSamples('/pscratch/sd/e/epaillas/planck/base_w/plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18/base_w_plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18',
                                    settings={'ignore_rows': 0.3})
samples_planck.addDerived(samples_planck.getParams().w, name='w0_fld', label='w_0')
print(samples_planck.getTable(limit=1).tableTex())
samples_list.append(samples_planck)

samples_labels = ['Planck TT,TE,EE+lowl+lowE', '+BAO+SNe', 'density-split + galaxy 2PCF']

g = plots.get_single_plotter(width_inch=6, ratio=4.5/5)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth = 2.0
g.settings.linewidth_contour = 3.0
g.settings.axes_fontsize = 23
g.settings.axes_labelsize = 26
g.settings.axis_tick_max_labels = 6
g.settings.line_styles = colors


g.plot_1d(samples_list, 'w0_fld',)
xmin, xmax = g.fig.axes[0].get_xlim()
g.fig.axes[0].fill_between([xmin, priors['w0_fld'][0]], 0.0, 1.0, color='grey', alpha=0.1)
g.fig.axes[0].fill_between([priors['w0_fld'][1], xmax], 0.0, 1.0, color='grey', alpha=0.1)
# plt.ylabel('Posterior density', fontsize=23)
plt.ylim(0.0, 1.65)
# plt.xlim(-1.2, -0.8)
g.add_legend(samples_labels, legend_loc='upper right', facecolor='w', fontsize=23)
# g.subplots[0, 0].fill_between([-2, 2], [-2.0, -2.0], [-0.628, -0.628], color='grey', alpha=0.3)
# g.subplots[0, 0].fill_between([-2.0, -1.22], [-0.628, -0.628], [0.621, 0.621], color='grey', alpha=0.3)
# g.subplots[0, 0].fill_between([-0.726, 2.0], [-0.628, -0.628], [0.621, 0.621], color='grey', alpha=0.3)
# plt.tight_layout()
plt.savefig('fig/pdf/cosmo_inference_cmass_w0.pdf', bbox_inches='tight')