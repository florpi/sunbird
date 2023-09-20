"""
Figure 8 (left panel): Cosmology inference from individual Nseries mocks
"""
from getdist import plots, MCSamples
import getdist
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sunbird.cosmology.growth_rate import Growth
getdist.chains.print_load_details = False


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
    loglikes = data[:, 0] * -1
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights, loglikes

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

root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')

phases = list(range(1, 85))
smin = 0.7
smax = 150

statistics = 'density_split_cross_density_split_auto_tpcf'
param_space = 'base'
names, labels = get_names_labels(param_space)

chain_handles = [
    f'nseries_cutsky_ph{i}_{statistics}_mae_patchycov_vol1_smin{smin:.2f}_smax{smax:.2f}_m02_q0134_{param_space}_bbn' for i in phases
]
chain_labels = [
    f'phase {i:03}' for i in phases
]

params_toplot = [
    'omega_cdm', 'sigma8_m', 'n_s',
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


# fig, ax = plt.subplots()
samples_list = []
means = []
bestfits = []
for i in range(len(chain_handles)):
    chain_fn = root_dir / chain_handles[i] / 'results.csv'
    chain, weights, loglikes = read_dynesty_chain(chain_fn)
    samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names, loglikes=loglikes, ranges=priors)
    samples_list.append(samples)
    bestfits.append([samples[name][-1] for name in names])
    means.append([samples.mean(name) for name in names])

means = np.asarray(means)
bestfits = np.asarray(bestfits)

samples_bestfits = MCSamples(samples=bestfits, names=names,
                    labels=labels, settings={'smooth_scale_2D': 0.95, 'smooth_scale_1D': 0.95})
samples_means = MCSamples(samples=means, names=names,
                    labels=labels)

systematic_error = {i: np.abs(samples_means.mean(i) - truth[i]) for i in params_toplot}
twosigma = {i: 2*samples_means.std(i)/np.sqrt(len(phases)) for i in params_toplot}
conservative = {i: max(systematic_error[i], twosigma[i]) for i in params_toplot}
print(f'number of samples: {len(phases)}')
print(f'systematic error: {systematic_error}')
print(f'2 sigma error: {twosigma}')
print(f'conservative error: {conservative}')

param_limits = {
    # 'omega_cdm': [0.112, 0.1293],
    'sigma8_m': [0.72, 0.9],
    # 'n_s': [0.9012, 1.0],
    # 'logsigma': [-0.7, -0.3],
    # 'alpha_s': [0.7, 1.0],
}

g = plots.get_subplot_plotter(width_inch=8)
g.settings.axis_marker_lw = 1.5
g.settings.axis_marker_ls = '-'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'crimson'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth_contour = 2.0
g.settings.linewidth=2.0
g.settings.legend_fontsize = 22
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 22
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.5
g.triangle_plot(roots=samples_bestfits,
                params=params_toplot,
                filled=False,
                legend_loc='upper right',
                markers=truth,
                param_limits=param_limits,
)

g.subplots[1, 0].scatter(bestfits[:, 1], bestfits[:, 2], color='grey', marker='o', s=50)
g.subplots[2, 0].scatter(bestfits[:, 1], bestfits[:, 3], color='grey', marker='o', s=50)
g.subplots[2, 1].scatter(bestfits[:, 2], bestfits[:, 3], color='grey', marker='o', s=50)

output_fn = f'fig/pdf/cosmo_inference_nseries.pdf'
plt.savefig(output_fn, bbox_inches='tight')