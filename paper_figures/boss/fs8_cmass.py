"""
Figure 5: fsigma8 comparison
"""
from getdist import MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sunbird.cosmology.growth_rate import Growth
from scipy.special import hyp2f1
import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

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

def fs8_of_z(z, s80=0.811, om_m = 0.308, gamma=0.55):
    
    Ez = np.sqrt((om_m * (1 + z) ** 3 + 1 - om_m))
    az = 1. / (1 + z)
    omega_l = 1. - om_m
    growth = az ** 2.5 * np.sqrt(omega_l + om_m * az ** (-3.)) * \
                  hyp2f1(5. / 6, 3. / 2, 11. / 6, -(omega_l * az ** 3.) / om_m) / \
                  hyp2f1(5. / 6, 3. / 2, 11. / 6, -omega_l / om_m)
    f = ((om_m * (1 + z)**3.) / (om_m * (1 + z)**3 + omega_l))**gamma
    
    return f * s80 * growth 


args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='/pscratch/sd/e/epaillas/sunbird/chains/enrique')
args.add_argument('--param_space', type=str, default='cosmo')
args = args.parse_args()

chain_dir = Path(args.chain_dir)

smin = 0.7
smax = 150
redshift = 0.525
growth = Growth(emulate=True,)
add_sys_error = True

chain_handles = [
    f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin{smin:.2f}_smax{smax:.2f}_m02_q0134_base_bbn_percival',
]
chain_labels = [
    'density-split clustering +\n galaxy 2PCF',
]

names = [
    'omega_b', 'omega_cdm', 'sigma8_m', 'n_s',
    # 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
    'logM1', 'logM_cut', 'alpha', 'alpha_s',
    'alpha_c', 'logsigma', 'kappa', 'B_cen', 'B_sat',
    'fsigma8'
]

labels = [
    r'\omega_{\rm b}',
    r'\omega_{\rm cdm}',
    r'\sigma_8',
    r'n_s',
    # r'\alpha_s',
    # r'N_{\rm ur}',
    # r'w_0',
    # r'w_a',
    r'\log M_1', r'\log M_{\rm cut}', r'\alpha', r'\alpha_{\rm vel, s}',
    r'\alpha_{\rm vel, c}', r'\log \sigma', r'\kappa', r'B_{\rm cen}', r'B_{\rm sat}',
    r'f\sigma_8',
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
param_limits = {
    'omega_cdm': [0.1128, 0.1295],
    'sigma8_m': [0.7, 0.9],
    'n_s': [0.895, 1.0],
    'logsigma': [-0.7, -0.3],
    'alpha_s': [0.7, 1.0],
}
sys_error = {
    'fsigma8': 0.006568666274607282,
}

samples_list = []

for i in range(len(chain_handles)):
    chain_fn = chain_dir / chain_handles[i] / 'results.csv'
    chain, weights = read_dynesty_chain(chain_fn, redshift=redshift)
    samples = MCSamples(samples=chain, weights=weights, labels=labels,
                        names=names, ranges=priors)
    samples_list.append(samples)
    print([f'{name} {samples.std(name) / samples.mean(name) * 100:.1f}' for name in names])
    fs8_mean = samples.mean('fsigma8')
    fs8_std = samples.std('fsigma8')
    print(samples.getTable(limit=1).tableTex())

    if add_sys_error:
        print(f'fsigma8 = {fs8_mean:.4f} +- {fs8_std:.4f} +- {sys_error["fsigma8"]:.4f}')
        fs8_std = np.sqrt(fs8_std**2 + sys_error['fsigma8']**2)


colors = ['royalblue', 'lightseagreen', 'crimson', 'orange']


plt.figure(figsize=(6.8, 3.5)) 
ms = 3
lw = 1.5

# plot Planck bands
zvals = np.linspace(0.0, 1.0)
om = 0.315; omerr = 0.007
s8 = 0.811; s8err = 0.006
fs8vals = fs8_of_z(zvals, om_m=om, s80=s8)
plt.plot(zvals, fs8vals, c='royalblue', lw=1.5, label='Planck 2018')
s = 2
fs8low = fs8_of_z(zvals, om_m=om-s*omerr, s80=s8-s*s8err)
fs8high = fs8_of_z(zvals, om_m=om+s*omerr, s80=s8+s*s8err)
plt.fill_between(zvals, fs8low, fs8high, alpha=0.2, color='royalblue')
s = 1
fs8low = fs8_of_z(zvals, om_m=om-s*omerr, s80=s8-s*s8err)
fs8high = fs8_of_z(zvals, om_m=om+s*omerr, s80=s8+s*s8err)
plt.fill_between(zvals, fs8low, fs8high, alpha=0.2, color='royalblue')

# plot literature measurements

kwargs = {'capsize': 1.5, 'elinewidth': 1.0, 'markeredgewidth': 1.0, 'ls':''}
lf = 0.1

plt.errorbar(0.55, fs8_mean, yerr=fs8_std, label='This work', marker='s', color=lighten_color('k', 1.0),
             ls='', markersize=5.0, capsize=1.5, elinewidth=1.0, markeredgewidth=1.0,
            markerfacecolor=lighten_color('crimson', 1.0), markeredgecolor=lighten_color('k', 1.0))

# plt.errorbar(0.54, 0.470, yerr=0.030, label='Galaxy 2PCF, this work', marker='s', color=lighten_color('k', 1.0),
#              ls='', markersize=5.0, capsize=1.5, elinewidth=1.0, markeredgewidth=1.0,
#             markerfacecolor=lighten_color('gold', 1.0), markeredgecolor=lighten_color('k', 1.0))

plt.errorbar([0.52], [0.444], yerr=[0.016], label='Yuan+22', marker='o', color='C1',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C1', **kwargs)

# plt.errorbar([0.57], [0.45], yerr=[0.01], label='Reid+14', marker='s', color='C2',
#             markersize=4.0, markerfacecolor=lighten_color('C2', 0.5), markeredgecolor='C2', **kwargs)
plt.errorbar([0.59], [0.455], yerr=[0.026], label='Yu+22', marker='>', color='C2',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C2', **kwargs)

plt.errorbar([0.25], [0.403], yerr=[0.03], label='Zhai+22', marker='*', color='C10',
            markersize=6.0, markerfacecolor=lighten_color('C10', lf), markeredgecolor='C10', **kwargs)

plt.errorbar([0.55 + 0.025], [0.399], yerr=[0.031], label="D'Amico+20", marker='d', color='C9',
            markersize=5.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C9', **kwargs)

plt.errorbar([0.4], [0.444], yerr=[0.025], marker='*', color='C10', markersize=6.0,
             markerfacecolor=lighten_color('w', lf), markeredgecolor='C10', **kwargs)

plt.errorbar([0.55], [0.385], yerr=[0.025], marker='*', color='C10', markersize=6.0,
             markerfacecolor=lighten_color('w', lf), markeredgecolor='C10', **kwargs)

plt.errorbar([0.24], [0.471], yerr=[0.024], label='Lange+21', marker='v', color='brown',
             markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='brown', **kwargs)

plt.errorbar([0.36], [0.431], yerr=[0.026], marker='v', color='brown', markersize=4.0,
                markerfacecolor=lighten_color('w', lf), markeredgecolor='brown', **kwargs)

plt.errorbar([0.38+0.005], [0.474], yerr=[0.031], label='Kobayashi+21', marker='^', color='C4',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C4', **kwargs)

plt.errorbar([0.61+0.02], [0.434], yerr=[0.027], marker='^', color='C4', markersize=4.0,
            markerfacecolor=lighten_color('w', lf), markeredgecolor='C4', **kwargs)

plt.errorbar([0.38 + 0.02], [0.497], yerr=[0.039], label='Alam+17', marker='<', color='C5',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C5', **kwargs)

plt.errorbar([0.51], [0.458], yerr=[0.035], marker='<', color='C5', markersize=4.0,
            markerfacecolor=lighten_color('w', lf), markeredgecolor='C5', **kwargs)

plt.errorbar([0.61], [0.436], yerr=[0.034], marker='<', color='C5', markersize=4.0,
            markerfacecolor=lighten_color('w', lf), markeredgecolor='C5', **kwargs)

plt.errorbar([0.9], [0.382], yerr=[0.06], label='de Mattia+20', marker='s', color='C6',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C6', **kwargs)

plt.errorbar([0.7], [0.451], yerr=[0.04], label='Bautista+20', marker='p', color='royalblue',
             markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='royalblue', **kwargs)

plt.errorbar([0.737], [0.408], yerr=[0.04], label='Chapman+21', marker='D', color='C7',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C7', **kwargs)

plt.errorbar([0.067], [0.423], yerr=[0.055], label='Beutler+12', marker='x', color='C8',
            markersize=4.0, markerfacecolor=lighten_color('w', lf), markeredgecolor='C8', **kwargs)


    # \item \cite{d'Amico2020:1909.05271}, $\Omega_m = 0.309 \pm 0.010$, $H_0 = 68.5 ± 2.2$, $f\sigma_8(z=0.55) = 0.399 \pm 0.031$

plt.xlabel('$z$', fontsize = 16)
plt.ylabel('$f(z)\sigma_8(z)$', fontsize = 16)
plt.legend(bbox_to_anchor=(1,1),fontsize=10.5, frameon=False)#ncol=2,)
plt.xlim(0.0, 1.0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('fig/pdf/fs8.pdf')