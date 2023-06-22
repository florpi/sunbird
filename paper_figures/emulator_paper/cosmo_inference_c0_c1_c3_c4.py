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

cosmologies = [0, 1, 3, 4]

# best hod for each cosmology
best_hod = {0: 26, 1:74, 3:30, 4:15}

chain_dir = Path(args.chain_dir)
chain_labels = [f'c{cosmo:03}' for cosmo in cosmologies]

params_names = [
    'omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
    'nrun', 'N_ur', 'w0_fld', 'wa_fld',
    'logM1', 'logM_cut', 'alpha', 'alpha_s',
    'alpha_c', 'sigma', 'kappa', 'B_cen', 'B_sat'
]

params_labels = [
    r'\omega_{\rm b}', r'\omega_{\rm cdm}', r'\sigma_8', r'n_s',
    r'\alpha_s', r'N_{\rm ur}', r'w_0', r'w_a',
    'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
    r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}'
]

params_toplot = [
    'omega_cdm', 'sigma8_m',
    # 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
    # 'logM1', 'logM_cut', 'alpha', 'alpha_s',
    # 'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat'
]

samples_list = []
markers = []
for i, cosmo in enumerate(cosmologies):
    hod = best_hod[cosmo]

    chain_handle = Path(f'abacus_cosmo{cosmo}_hod{hod}_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134')
    true_params = get_true_parameters(params_toplot, cosmo=cosmo, hod=hod)
    markers.append(true_params)

    chain_fn = chain_dir / chain_handle / 'results.csv'
    data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    samples = MCSamples(samples=chain, weights=weights, labels=params_labels, names=params_names)
    samples_list.append(samples)

g = plots.get_subplot_plotter(width_inch=6)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = '-'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 22
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 22
g.settings.axis_tick_x_rotation = 0
g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = colors
g.triangle_plot(roots=samples_list,
                params=params_toplot,
                filled=True,
                legend_labels=chain_labels,
                legend_loc='upper right',
)

for i, sample in enumerate(samples_list):
    markers_sample = markers[i]
    marker_color = colors[:len(samples_list)][::-1][i]
    for p1_idx, p1 in enumerate(params_toplot):
        marker1 = markers_sample[p1_idx]
        for p2_idx in range(p1_idx, len(params_toplot)):
            marker2 = markers_sample[p2_idx]
            ax = g.subplots[p2_idx, p1_idx]
            g.add_x_marker(marker1, ax=ax, 
                            color=marker_color, 
                            linewidth=3, linestyle='-',
            )
            if p1_idx != p2_idx:
                g.add_y_marker(marker2, ax=ax, 
                                color=marker_color, 
                                linewidth=3, linestyle='-',)
                ax.plot(marker1, marker2, color=marker_color, marker='s', alpha=0.75
                )

cosmo_str = '_'.join([f'c{cosmo}' for cosmo in cosmologies])
output_fn = f'fig/cosmo_inference_{cosmo_str}.pdf'
plt.savefig(output_fn, bbox_inches='tight')
plt.show()