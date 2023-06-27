from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def read_dynesty_chain(filename):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights


cosmo = 0

if cosmo == 0:
    hod = 26
elif cosmo == 1:
    hod = 74
elif cosmo == 3:
    hod = 30
elif cosmo == 4:
    hod = 15

root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/enrique')

phases = list(range(0, 25))

statistics = 'density_split_cross_density_split_auto_tpcf'

chain_handles = [
    f'abacus_cutsky_ph{i}_{statistics}_mae_patchycov_smin0.70_smax150.00_m02_q0134_basecosmo' for i in phases
]
chain_labels = [
    f'phase {i:03}' for i in phases
]

names = [
    'omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
    'logM1', 'logM_cut', 'alpha', 'alpha_s',
    'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat'
]
labels = [
    r'\omega_{\rm b}',
    r'\omega_{\rm cdm}',
    r'\sigma_8',
    r'n_s',
    'logM_1', r'logM_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
    r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}'
]

params_toplot = [
    'omega_cdm', 'sigma8_m',
    # 'logM1', 'logM_cut', 'alpha', 'alpha_s',
    # 'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat'
]

params = {
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "sigma8_m": 0.807952,
    "n_s": 0.9649,
    "nrun": 0.0,
    "N_ur": 2.0328,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
}
truth = [params[i] for i in params_toplot]

fig, ax = plt.subplots()
samples_list = []
means = []
for i in range(len(chain_handles)):
    chain_fn = root_dir / chain_handles[i] / 'results.csv'
    data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
    samples_list.append(samples)
    ax.scatter(samples.mean('omega_cdm'), samples.mean('sigma8_m'), color='grey')
    means.append([samples.mean('omega_cdm'), samples.mean('sigma8_m')])

means = np.asarray(means)
mean_of_means = np.mean(means, axis=0)
std_of_means = np.std(means, axis=0) #/ np.sqrt(len(means))

ax.errorbar(mean_of_means[0], mean_of_means[1], xerr=std_of_means[0], yerr=std_of_means[1],
            c='k', marker='*', ms=15, capsize=5,)

ax.plot(truth[0], truth[1], ls='', ms=15, marker='x', color='crimson',)

ax.set_xlabel(r'$\omega_{\rm cdm}$', fontsize=17)
ax.set_ylabel(r'$\sigma_8$', fontsize=17)
ax.tick_params(axis='x', labelsize=13, rotation=45)
ax.tick_params(axis='y', labelsize=13)
plt.show()

# mock_idx = np.random.randint(0, len(samples_list), size=5)
mock_idx = [0, 1, 2, 3, 4]


g = plots.get_subplot_plotter(width_inch=8)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
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
# g.settings.solid_colors = colors
g.triangle_plot(roots=[samples_list[i] for i in mock_idx],
                params=params_toplot,
                filled=True,
                legend_labels=[chain_labels[i] for i in mock_idx],
                legend_loc='upper right',
                # title_limit=1,
                markers=truth,
)

# plt.savefig('fig/mae_vs_learned_gaussian.pdf')
# plt.savefig('tpcf_boss.png', dpi=300)
# plt.savefig('test_avg_los.pdf')
plt.savefig('fig/cosmo_inference_abacus_cutsky.pdf')
# g.fig.show()