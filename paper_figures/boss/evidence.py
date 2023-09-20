"""
Figure 7: Bayesian evidence
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sunbird.data.data_readers import NseriesCutsky, CMASS
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import Bundle
from getdist import MCSamples
plt.style.use(['stylelib/science.mplstyle'])


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
    labels = [labels_dict[name] for name in names]
    return names, labels

fig, ax = plt.subplots(figsize=(6, 5))
colors = ['lightseagreen', 'lightpink', 'darkorchid']

loglikes_smin = []
evidence_smin = []
for i, smin in enumerate([0.7, 50.0]):
    statistic = ['density_split_cross', 'density_split_auto', 'tpcf']
    s = np.load('/pscratch/sd/e/epaillas/sunbird/data/s.npy')
    smax = 150
    s = s[(s > smin) & (s < smax)]
    quantiles = [0, 1, 3, 4]
    slice_filters = {'s': [smin, smax],}
    select_filters = {'quintiles': quantiles, 'multipoles': [0, 2],}
    loglikes_phases = []
    evidence_phases = []

    for phase in range(1, 84):
        print(phase)

        root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')
        chain_handle = f'nseries_cutsky_ph{phase}_density_split_cross_density_split_auto_tpcf_mae_patchycov_vol1_smin{smin:.2f}_smax150.00_m02_q0134_base_bbn'

        names, labels = get_names_labels(chain_handle)
        chain_fn = root_dir / chain_handle / 'results.csv'
        data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
        chain = data[:, 4:]
        loglikes = data[:, 1]
        evidence = data[:, 2]
        weights = np.exp(data[:, 1] - data[-1, 2])
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)

        # this reads the ML point from the chain
        parameters = {names[i]: samples[names[i]][-1] for i in range(len(names))}
        parameters['nrun'] = 0.0
        parameters['N_ur'] = 2.0328
        parameters['w0_fld'] = -1.0
        parameters['wa_fld'] = 0.0

        datavector = NseriesCutsky(
            statistics=statistic,
            select_filters=select_filters,
            slice_filters=slice_filters,
        ).get_observation(phase=phase)

        cov = CovarianceMatrix(
            covariance_data_class='Patchy',
            statistics=statistic,
            select_filters=select_filters,
            slice_filters=slice_filters,
            path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/'
        )

        emulator = Bundle(
            summaries=statistic,
            path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/',
        )

        model, error_model = emulator(
            param_dict=parameters,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )

        cov_data = cov.get_covariance_data(volume_scaling=1)
        cov_emu = cov.get_covariance_emulator()
        cov_sim = cov.get_covariance_simulation()
        cov_tot = cov_data + cov_emu + cov_sim
        error_data = np.sqrt(np.diag(cov_data))
        error_emu = np.sqrt(np.diag(cov_emu))
        error_sim = np.sqrt(np.diag(cov_sim))
        error_model = np.sqrt(error_sim**2 + error_emu**2)
        error_tot = np.sqrt(error_data**2 + error_emu**2 + error_sim**2)

        dof = (len(datavector) - 13)
        chi2 = np.dot(datavector - model, np.linalg.inv(cov_tot)).dot(datavector - model)
        chi2_red = chi2 / dof
        # print(f'{statistic} reduced chi2 = {chi2_red}')
        # chi2_list.append(chi2_red)
        loglikes_phases.append(loglikes[-1])
        evidence_phases.append(evidence[-1])


    root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/boss_paper')
    chain_handle = f'cmass_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin{smin:.2f}_smax150.00_m02_q0134_base_bbn'

    names, labels = get_names_labels(chain_handle)
    chain_fn = root_dir / chain_handle / 'results.csv'
    data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    loglikes = data[:, 1]
    evidence = data[:, 2]
    weights = np.exp(data[:, 1] - data[-1, 2])
    samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)

    # this reads the ML point from the chain
    parameters = {names[i]: samples[names[i]][-1] for i in range(len(names))}
    parameters['nrun'] = 0.0
    parameters['N_ur'] = 2.0328
    parameters['w0_fld'] = -1.0
    parameters['wa_fld'] = 0.0

    datavector = CMASS(
        statistics=statistic,
        select_filters=select_filters,
        slice_filters=slice_filters,
        region='NGC'
    ).get_observation()

    cov = CovarianceMatrix(
        covariance_data_class='Patchy',
        statistics=statistic,
        select_filters=select_filters,
        slice_filters=slice_filters,
        path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/'
    )

    emulator = Bundle(
        summaries=statistic,
        path_to_models='/global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/best/',
    )

    model, error_model = emulator(
        param_dict=parameters,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )

    cov_data = cov.get_covariance_data()
    cov_emu = cov.get_covariance_emulator()
    cov_sim = cov.get_covariance_simulation()
    cov_tot = cov_data + cov_emu + cov_sim

    dof = (len(datavector) - 13)
    chi2 = np.dot(datavector - model, np.linalg.inv(cov_tot)).dot(datavector - model)
    chi2_red = chi2 / dof

    ax.hist(evidence_phases, bins=20, alpha=0.5, color=colors[i],)
    ylim = ax.get_ylim()
    ax.vlines(evidence[-1], 0, 10, color=colors[i], linestyle='--',)


ax.plot(np.nan, np.nan, ls='-', lw=7.0, color='k',
           label='Nseries',)
ax.plot(np.nan, np.nan, ls='--', lw=2.0, color='k',
           label='CMASS',)

ax.annotate(text=r'$s_{\rm min} = 1\,{h^{-1}{\rm Mpc}}$', xy=(0.08, 0.87),
               xycoords='axes fraction', fontsize=14, color=colors[0])
ax.annotate(text=r'$s_{\rm min} = 50\,{h^{-1}{\rm Mpc}}$', xy=(0.6, 0.87),
               xycoords='axes fraction', fontsize=14, color=colors[1])

leg = ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center',
        frameon=False, fontsize=15, ncols=2, columnspacing=0.7,)

for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

ax.set_ylim(0, 13.0)
ax.set_xlabel('log-evidence 'r'$\mathcal{Z}$', fontsize=15)
ax.set_ylabel(r'counts', fontsize=15)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig('fig/pdf/evidence.pdf', bbox_inches='tight')