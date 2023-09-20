"""
Figure 8 (right panel): Cosmology inference from mean of Nseries mocks
"""
from getdist import plots, MCSamples
import getdist
import pandas as pd
from pathlib import Path
from sunbird.cosmology.growth_rate import Growth
import numpy as np
import matplotlib.pyplot as plt
getdist.chains.print_load_details = False


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

cosmo = 0

if cosmo == 0:
    hod = 26
elif cosmo == 1:
    hod = 74
elif cosmo == 3:
    hod = 30
elif cosmo == 4:
    hod = 15

root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')

phases = list(range(0, 1))
smin = 0.7
smax = 150

statistics = 'density_split_cross_density_split_auto_tpcf'
param_space = 'base'
names, labels = get_names_labels(param_space)

chain_handles = [
    f'nseries_cutsky_ph0_{statistics}_mae_patchycov_vol1_smin{smin:.2f}_smax{smax:.2f}_m02_q0134_{param_space}_bbn',
    f'nseries_cutsky_ph0_{statistics}_mae_patchycov_vol84_smin{smin:.2f}_smax{smax:.2f}_m02_q0134_{param_space}_bbn',
]
chain_labels = [
    'Nseries, 'r'$V = 1.4\,(h^{-3}{\rm Gpc}^3)$',
    'Nseries, 'r'$V = 120\,(h^{-3}{\rm Gpc}^3)$',
]


params_toplot = [
    'omega_cdm', 'sigma8_m', 'n_s',
    # 'logM1', 'logM_cut', 'alpha', 'alpha_s',
    # 'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat'
]

# Nseries cosmology
Omega_m = 0.286
Omega_b = 0.047
Omega_cdm = Omega_m - Omega_b
h = 0.7
omega_cdm = Omega_cdm * h**2
omega_b = Omega_b * h**2
truth = {
    "omega_b": 0.02237,
    "omega_cdm": omega_cdm,
    "sigma8_m": 0.82,
    "n_s": 0.96,
    "nrun": 0.0,
    "N_ur": 2.0328,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
}

redshift = 0.525
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
truth['fsigma8'] = pred_fsigma8[0][0]

samples_list = []
means = []
for i in range(len(chain_handles)):
    chain_fn = root_dir / chain_handles[i] / 'results.csv'
    names, labels = get_names_labels(chain_handles[i])
    chain, weights = read_dynesty_chain(chain_fn)
    samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
    print(samples.getTable(limit=2).tableTex())
    systematic_error = {i: np.abs(samples.mean(i) - truth[i]) for i in params_toplot}
    print(f'systematic error: ', [f"{key}: {value:.5f}" for key, value in systematic_error.items()])
    samples_list.append(samples)


g = plots.get_subplot_plotter(width_inch=8)
g.settings.axis_marker_lw = 1.5
g.settings.axis_marker_ls = '-'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'crimson'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth_contour = 2.0
g.settings.linewidth = 2.0
g.settings.legend_fontsize = 21
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 22
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.5
g.settings.line_styles = [('--', 'grey'), ('-', 'k')]
g.triangle_plot(roots=samples_list,
                params=params_toplot,
                filled=False,
                legend_loc='upper right',
                legend_labels=chain_labels,
                markers=truth,
)
output_fn = f'fig/pdf/cosmo_inference_nseries_mean.pdf'
plt.savefig(output_fn, bbox_inches='tight')