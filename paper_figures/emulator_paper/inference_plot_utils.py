import numpy as np
import pandas as pd
from cosmoprimo.fiducial import AbacusSummit
from getdist import plots, MCSamples
from sunbird.cosmology.growth_rate import Growth
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

labels = {
    "omega_b": r"\omega_{\rm b}",
    "omega_cdm": r"\omega_{\rm cdm}",
    "sigma8_m": r"\sigma_8",
    "n_s": r"n_s",
    "nrun": r"{\rm d}n_s/{\rm d}\ln k",
    "N_eff": r"N_{\rm eff}",
    "N_ur": r"N_{\rm ur}",
    "w0_fld": r"w_0",
    "wa_fld": r"w_a",
    "logM1": "logM_1",
    "logM_cut": r"logM_{\rm cut}",
    "alpha": r"\alpha",
    "alpha_s": r"\alpha_{\rm vel, s}",
    "alpha_c": r"\alpha_{\rm vel, c}",
    "logsigma": r"\log \sigma",
    "kappa": r"\kappa",
    "B_cen": r"B_{\rm cen}",
    "B_sat": r"B_{\rm sat}",
    "fsigma8": r"$f \sigma_8",
}


def read_dynesty_chain(filename, add_fsigma8=False, redshift=0.5):
    data = pd.read_csv(filename)
    param_names = list(data.columns[4:])
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        data["fsigma8"] = growth.get_fsigma8(
            omega_b=data["omega_b"].to_numpy(),
            omega_cdm=data["omega_cdm"].to_numpy(),
            sigma8=data["sigma8_m"].to_numpy(),
            n_s=data["n_s"].to_numpy(),
            N_ur=data["N_ur"].to_numpy(),
            w0_fld=data["w0_fld"].to_numpy(),
            wa_fld=data["wa_fld"].to_numpy(),
            z=redshift,
        )
        param_names.append("fsigma8")
    data["N_eff"] = data["N_ur"] + 1.0132
    param_names.append("N_eff")
    data = data.to_numpy()
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return param_names, chain, weights


def read_hmc_chain(filename, add_fsigma8=False, redshift=0.5):
    data = pd.read_csv(filename)
    param_names = list(data.columns)
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        data["fsigma8"] = growth.get_fsigma8(
            omega_b=data["omega_b"].to_numpy(),
            omega_cdm=data["omega_cdm"].to_numpy(),
            sigma8=data["sigma8_m"].to_numpy(),
            n_s=data["n_s"].to_numpy(),
            N_ur=data["N_ur"].to_numpy(),
            w0_fld=data["w0_fld"].to_numpy(),
            wa_fld=data["wa_fld"].to_numpy(),
            z=redshift,
        )
        param_names.append("fsigma8")
    data["N_eff"] = data["N_ur"] + 1.0132
    param_names.append("N_eff")
    data = data.to_numpy()
    return param_names, data, None


def get_MCSamples(
    filename,
    add_fsigma8=False,
    redshift=0.5,
    hmc=True,
):
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
        "N_eff": [2.1902, 3.9022],
    }
    if hmc:
        names, chain, weights = read_hmc_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )
    else:
        names, chain, weights = read_dynesty_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )

    samples = MCSamples(
        samples=chain,
        weights=weights,
        labels=[labels[n] for n in names],
        names=names,
        ranges=priors,
    )
    print(samples.getTable(limit=1).tableTex())
    return samples


def get_true_params(
    cosmology,
    hod_idx,
    add_fsigma8=False,
    redshift=0.5,
):
    param_dict = dict(
        pd.read_csv(
            f"../../data/parameters/abacus/bossprior/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        ).iloc[hod_idx]
    )
    if add_fsigma8:
        cosmo = AbacusSummit(cosmology)
        param_dict["fsigma8"] = cosmo.sigma8_z(redshift) * cosmo.growth_rate(redshift)
    param_dict["N_eff"] = param_dict["N_ur"] + 1.0132
    return param_dict


def plot_corner(
    samples_list,
    params_to_plot,
    chain_labels,
    true_params,
    inches_per_param=9 / 7,
    markers=None,
    markers_colors=["lightgray"],
    colors=["#4165c0", "#e770a2", "#5ac3be", "#696969", "#f79a1e", "#ba7dcd"],
):
    g = plots.get_subplot_plotter(width_inch=inches_per_param * len(params_to_plot))
    g.settings.constrained_layout = True
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_ls = ":"
    g.settings.title_limit_labels = False
    g.settings.axis_marker_color = "k"
    g.settings.legend_colored_text = True
    g.settings.figure_legend_frame = False
    g.settings.linewidth_contour = 1.0
    g.settings.legend_fontsize = 22
    g.settings.axes_fontsize = 16
    g.settings.axes_labelsize = 20
    g.settings.axis_tick_x_rotation = 45
    # g.settings.axis_tick_y_rotation = 45
    g.settings.axis_tick_max_labels = 6
    g.settings.solid_colors = colors
    g.triangle_plot(
        roots=samples_list,
        params=params_to_plot,
        filled=True,
        legend_labels=chain_labels,
        legend_loc="upper right",
        markers=true_params,
    )
    if markers is not None:
        for i, marker in enumerate(markers):
            for p1_idx, p1 in enumerate(params_to_plot):
                if p1 in marker:
                    marker1 = marker[p1]
                    for p2_idx in range(p1_idx, len(params_to_plot)):
                        p2 = params_to_plot[p2_idx]
                        marker2 = marker[p2]
                        ax = g.subplots[p2_idx, p1_idx]
                        g.add_x_marker(
                            marker1,
                            ax=ax,
                            color=markers_colors[i],
                            lw=1.25,
                            ls="-",
                            alpha=0.6,
                        )
                        if p1_idx != p2_idx:
                            g.add_y_marker(
                                marker2,
                                ax=ax,
                                color=markers_colors[i],
                                lw=1.25,
                                ls="-",
                                alpha=0.6,
                            )
                            ax.plot(
                                marker1,
                                marker2,
                                color=markers_colors[i],
                                marker="s",
                                alpha=0.6,
                            )
    return g
