import json
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from sunbird.data.data_readers import Abacus 
from sunbird.cosmology.growth_rate import Growth
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science', 'vibrant'])


labels_dict = {
    "omega_b": r'\omega_{\rm b}', "omega_cdm": r'\omega_{\rm cdm}',
    "sigma8_m": r'\sigma_8', "n_s": r'n_s', "nrun": r'\alpha_s',
    "N_ur": r'N_{\rm ur}', "w0_fld": r'w_0', "wa_fld": r'w_a',
    "logM1": r'\log M_1', "logM_cut": r'\log M_{\rm cut}',
    "alpha": r'\alpha', "alpha_s": r'\alpha_{\rm vel, s}',
    "alpha_c": r'\alpha_{\rm vel, c}', "logsigma": r'\log \sigma',
    "kappa": r'\kappa', "B_cen": r'B_{\rm cen}', "B_sat": r'B_{\rm sat}',
    "fsigma8": r'f \sigma_8' ,
}
def get_fsigma8(params, redshift=0.5,):
    growth = Growth(
        emulate=True,
    )
    return growth.get_fsigma8(
        omega_b=params["omega_b"].to_numpy(),
        omega_cdm=params["omega_cdm"].to_numpy(),
        sigma8=params["sigma8_m"].to_numpy(),
        n_s=params["n_s"].to_numpy(),
        N_ur=params["N_ur"].to_numpy(),
        w0_fld=params["w0_fld"].to_numpy(),
        wa_fld=params["wa_fld"].to_numpy(),
        z=redshift,
    )

def read_hmc_chain(filename, add_fsigma8=True, redshift=0.5,):
    data = pd.read_csv(filename)
    param_names = list(data.columns)
    if add_fsigma8:
        data['fsigma8'] = get_fsigma8(data, redshift=redshift)
        param_names.append("fsigma8")
    data = data.to_numpy()
    return param_names, data 

def get_ranks(
    posterior_samples_array,
    theta,
):
    n_posteriors = theta.shape[0]
    ndim = theta.shape[1]
    ranks, mus, stds = [], [], []
    for i in range(n_posteriors):
        posterior_samples = posterior_samples_array[i]
        mu, std = posterior_samples[i].mean(axis=0), posterior_samples.std(axis=0)
        rank = [(posterior_samples[:, j] < theta[i, j]).sum() for j in range(ndim)]
        mus.append(mu)
        stds.append(std)
        ranks.append(rank)
    mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
    return mus, stds, ranks

def plot_coverage(ranks, labels, plotscatter=True):
    ncounts = ranks.shape[0]
    npars = ranks.shape[-1]
    unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(30)]

    fig, ax = plt.subplots(figsize=(3.5,2.6),)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = cmap(np.linspace(0.01, 0.99, len(labels)))

    for i in range(npars):
        xr = np.sort(ranks[:, i])
        xr = xr / xr[-1]
        cdf = np.arange(xr.size) / xr.size
        ax.plot(xr, cdf, lw=1.25, label=labels[i], color=colors[i])
    ax.set_xlabel('Confidence Level')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    ax.set_ylabel('Empirical Coverage')
    ax.text(0.1, 0.85, 'Underconfident', ha='left', va='top', transform=ax.transAxes, style='italic',fontsize=9)
    ax.text(0.9, 0.15, 'Overconfident', ha='right', va='bottom', transform=ax.transAxes, style='italic',fontsize=9)
    if plotscatter:
        for j in range(len(unicov)): ax.plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
    return ax

if __name__ == '__main__':
    # Read chains
    args = argparse.ArgumentParser()
    args.add_argument(
        "--chain_dir",
        type=str,
        default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper/",
    )
    args.add_argument(
        "--loss",
        type=str,
        default='learned_gaussian',
    )
    args = args.parse_args()

    loss_value = args.loss
    data_path = Path(args.chain_dir)
    abacus = Abacus(
        statistics=['tpcf'],
    )
    with open("../../data/train_test_split.json") as f:
        train_test_split = json.load(f)
    test_cosmologies = train_test_split['test']
    hod_range = range(100)

    params_to_plot = ['omega_cdm', 'sigma8_m', 'n_s', 'fsigma8', 'logM_cut', 'B_cen']

    posterior_samples, theta = [], []
    for cosmology in test_cosmologies:
        true_parameters = abacus.get_all_parameters(
            cosmology=cosmology,
        )
        true_parameters['fsigma8'] = get_fsigma8(true_parameters)
        true_parameters = true_parameters[params_to_plot]
        theta.append(true_parameters)
        for hod_idx in hod_range:
            filename = f'cos={cosmology}-h={hod_idx}-o=Abacus-l={loss_value}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1'
            # read chain
            param_names, samples = read_hmc_chain(data_path / filename / 'results.csv')
            indices = [param_names.index(col) for col in true_parameters.columns]
            samples = samples[:, indices]
            param_names = [param_names[idx] for idx in indices]
            posterior_samples.append(samples)
    # Reorder parameters in the same way
    posterior_samples = np.array(posterior_samples)
    theta = np.array(theta)
    theta = theta.reshape(-1, theta.shape[-1])

    mus, stds, ranks = get_ranks(posterior_samples, theta)
    ax = plot_coverage(ranks=ranks,labels=['$' + labels_dict[param] + '$' for param in param_names],)
    plt.savefig('figures/png/F8_coverage.png', dpi=300)
    plt.savefig('figures/pdf/F8_coverage.pdf', dpi=300)