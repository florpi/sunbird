from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse


colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def read_dynesty_chain(filename):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights

def get_true_parameters(params_toplot, cosmo=0, hod=26):
    params_dir = f'../../data/parameters/abacus/bossprior/AbacusSummit_c{cosmo:03}.csv'
    params = pd.read_csv(params_dir)
    params = [params[i][hod] for i in params_toplot]
    return params


args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='../../chains/enrique')
args = args.parse_args()

chain_dir = Path(args.chain_dir)

# best hod for each cosmology
best_hod = {0: 26, 1:74, 3:30, 4:15}

cosmo = 0
hod = best_hod[cosmo]

chain_handles = [
    f'abacus_cosmo{cosmo}_hod{hod}_tpcf_mae_vol64_smin0.70_smax150.00_m02_q0134',
    f'abacus_cosmo{cosmo}_hod{hod}_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134',
]
chain_labels = [
    'Galaxy 2PCF',
    'Density-split clustering',
]

names = [
    'omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
    'nrun', 'N_ur', 'w0_fld', 'wa_fld',
    'logM1', 'logM_cut', 'alpha', 'alpha_s',
    'alpha_c', 'sigma', 'kappa', 'B_cen', 'B_sat'
]
labels = [
    r'\omega_{\rm b}', r'\omega_{\rm cdm}', r'\sigma_8', r'n_s',
    r'\alpha_s', r'N_{\rm ur}', r'w_0', r'w_a',
    'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
    r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}'
]

params_toplot = [
    'omega_cdm', 'sigma8_m', 'n_s',
    'nrun', 'N_ur', 'w0_fld', 'wa_fld',
    # 'logM1', 'logM_cut', 'alpha', 'alpha_s',
    # 'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat'
]

true_params = get_true_parameters(params_toplot, cosmo=cosmo, hod=hod)

samples_list = []
for i in range(len(chain_handles)):
    chain_fn = chain_dir / chain_handles[i] / 'results.csv'
    data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
    samples_list.append(samples)
    print(samples.getTable(limit=1).tableTex())

g = plots.get_subplot_plotter(width_inch=8)
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 21
g.settings.axes_fontsize = 16
g.settings.axes_labelsize = 21
g.settings.axis_tick_x_rotation = 45
# g.settings.axis_tick_y_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = colors
g.triangle_plot(roots=samples_list,
                params=params_toplot,
                filled=True,
                legend_labels=chain_labels,
                legend_loc='upper right',
                # title_limit=1,
                markers=true_params,
)
plt.tight_layout()
plt.savefig('fig/cosmo_inference_c0_hod26.pdf', bbox_inches='tight')