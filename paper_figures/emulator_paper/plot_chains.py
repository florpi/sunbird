import pandas as pd
from pathlib import Path
import numpy as np
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
plt.style.use(["science"])


labels = {
    "omega_b": r"\omega_{\rm b}",
    "omega_cdm": r"\omega_{\rm cdm}",
    "sigma8_m": r"\sigma_8",
    "n_s": "n_s",
    "nrun": r"\alpha_s",
    "N_ur": r"N_\mathrm{ur}",
    "w0_fld": r"w_0",
    "wa_fld": r"w_a",
    "logM1": r"\log M_1",
    "logM_cut": r"\log M_\mathrm{cut}",
    "alpha": r"\alpha",
    "alpha_s": r"\alpha_\mathrm{vel,s}",
    "alpha_c": r"\alpha_\mathrm{vel,c}",
    "logsigma": r"\log \sigma",
    "kappa": r"\kappa",
    "B_cen": r"B_\mathrm{cen}",
    "B_sat": r"B_\mathrm{sat}",
}


def get_true_params(cosmology, hod_idx):
    return dict(
        pd.read_csv(
            f"../../data/parameters/abacus/bossprior/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        ).iloc[hod_idx]
    )


def get_weights(df):
    return np.exp(df["log_weights"] - df["log_evidence"].iloc[-1]).values


def get_chain(chain_path, cosmology, hod_idx, suffix=None):
    file_path = f"abacus_cosmo{cosmology}_hod{hod_idx}"
    if suffix is not None:
        file_path += f"_{suffix}"
    return get_chain_from_full_path(
        chain_path / f"{file_path}/results.csv",
    )


def get_chain_from_full_path(full_path):
    df = pd.read_csv(full_path)
    params = [
        c
        for c in df.columns
        if c
        not in ("log_likelihood", "log_weights", "log_evidence", "log_evidence_err")
    ]
    weights = get_weights(df)
    return MCSamples(
        samples=df[params].values,
        labels=[labels[p] for p in params],
        names=params,
        weights=weights,
        loglikes=df["log_likelihood"].values,
    )




def get_chain_from_full_path_hmc(full_path):
    df = pd.read_csv(full_path)
    params = [
        c
        for c in df.columns
        if c
        not in ("log_likelihood", "log_weights", "log_evidence", "log_evidence_err")
    ]
    return MCSamples(
        samples=df[params].values,
        labels=[labels[p] for p in params],
        names=params,
    )
def plot_samples(
    samples_list,
    true_params,
    params,
    colors,
    labels,
    markers=None,
    markers_colors=None,
    g=None,
):
    if g is None:
        g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 22
    g.settings.legend_fontsize = 25
    g.settings.axes_labelsize = 23
    g.settings.axis_marker_lw = 1.5
    g.settings.title_limit_fontsize = 20
    g.settings.title_limit_labels = False
    g.settings.tight_layout = True
    g.settings.axis_marker_color = "dimgrey"
    g.settings.legend_colored_text = True
    g.settings.figure_legend_frame = False
    g.triangle_plot(
        roots=samples_list,
        params=params,
        filled=True,
        legend_labels=labels,
        legend_loc="upper right",
        line_args=[{"lw": 2, "color": color} for color in colors],
        contour_colors=colors,
        title_limit=1,
        markers=[true_params[c] for c in params] if true_params is not None else None,
    )
    if markers is not None:
        for i, marker in enumerate(markers):
            for p1_idx, p1 in enumerate(params):
                marker1 = marker[p1]
                for p2_idx in range(p1_idx, len(params)):
                    p2 = params[p2_idx]
                    marker2 = marker[p2]
                    ax = g.subplots[p2_idx, p1_idx]
                    g.add_x_marker(
                        marker1,
                        ax=ax,
                        color=markers_colors[i],
                        linewidth=3,
                        linestyle="--",
                        alpha=0.75,
                    )
                    if p1_idx != p2_idx:
                        g.add_y_marker(
                            marker2,
                            ax=ax,
                            color=markers_colors[i],
                            linewidth=3,
                            linestyle="--",
                            alpha=0.75,
                        )
                        ax.plot(
                            marker1,
                            marker2,
                            color=markers_colors[i],
                            marker="s",
                        )
    return g

if __name__ == '__main__':
    path_to_chains = Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper')

    statistics = [
        #'nested_density_split_cross_nopreduncertainty_gauss',
        #'density_split_cross_gauss',
        'nested_density_split_cross_gauss_nopreduncertainty',
        'nested_density_split_cross_mae_nopreduncertainty',
        #'nested_density_split_cross_nopreduncertainty_mae', 
    ]
    filenames = [
        f'bossprior_smin0.70_smax150.00_q0134_m02_{statistics[0]}',
        f'bossprior_smin0.70_smax150.00_q0134_m02_{statistics[1]}',
    ]
    chains = [
        #get_chain_from_full_path(path_to_chains / f'{filenames[0]}/results.csv'),
        #get_chain_from_full_path(path_to_chains / f'{filenames[1]}/results.csv'),
        get_chain_from_full_path(path_to_chains.parent / 'enrique/abacus_cosmo0_hod26_smin0.70_smax150.00_q0134_m02_ds_cross_vol64_test2/results.csv'),
        #get_chain_from_full_path(path_to_chains / 'bossprior_cross/results.csv'),
    ]
    true_params = get_true_params(0,26)
    g = plot_samples(
        chains,
        true_params=None,
        params=[
            'omega_b',
            'omega_cdm', 
            'sigma8_m',  
            'n_s', 
            "nrun",
            "N_ur",
            "w0_fld",
            "wa_fld",
        ],
        colors=['lightseagreen', 'mediumorchid',],# 'blue', 'gray'],
        #labels=statistics,
        #labels=['TPCF', r'$\mathrm{DS}_\mathrm{cross}$',],
        labels=['Enrique',],# 'Mine'],
        markers=[
            true_params,
        ],
        markers_colors=['lightgray'],
    )
    plt.savefig(f"figures/png/inference.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/inference.pdf", bbox_inches="tight")


    g = plot_samples(
        chains,
        true_params=None,
        params=[
            "logM1",
            "logM_cut",
            "alpha",
            "alpha_s",
            "alpha_c",
            "logsigma",
            "kappa",
            "B_cen",
            "B_sat",
        ],
        colors=['lightseagreen', 'mediumorchid',],# 'blue', 'gray'],
        #labels=statistics,
        #labels=['TPCF', r'$\mathrm{DS}_\mathrm{cross}$',],
        labels=['Enrique', 'Mine'],
        markers=[
            true_params,
        ],
        markers_colors=['lightgray'],
    )
    plt.savefig(f"figures/png/inference_hod.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/inference_hod.pdf", bbox_inches="tight")
