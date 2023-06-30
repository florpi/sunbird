from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots


args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/enrique",
)
args = args.parse_args()

chain_dir = Path(args.chain_dir)
# best hod for each cosmology
best_hod = {0: 26, 1: 74, 3: 30, 4: 15}

cosmo = 0
hod = best_hod[cosmo]

chain_handles = [
    f"abacus_cosmo{cosmo}_hod{hod}_tpcf_mae_vol64_smin0.70_smax150.00_m02_q0134",
    f"abacus_cosmo{cosmo}_hod{hod}_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134",
]
chain_labels = [
    "AbacusSummit 2PCF",
    "AbacusSummit Density-Split",
]

cosmo_params = [
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "nrun",
    "N_ur",
    "w0_fld",
    "wa_fld",
]
hod_params = [
    "logM1",
    "logM_cut",
    "alpha",
    "alpha_s",
    "alpha_c",
    "logsigma",
    "kappa",
    "B_cen",
    "B_sat",
]

true_params = inference_plots.get_true_params(cosmo, hod)

samples_list = []
for chain_handle in chain_handles:
    samples_list.append(
        inference_plots.get_MCSamples(
            chain_dir / chain_handle / "results.csv",
        )
    )

ax = inference_plots.plot_corner(
    samples_list,
    cosmo_params,
    chain_labels,
    true_params=None,
    markers=[true_params],
)
plt.savefig("figures/pdf/cosmo_inference_c0_hod26.pdf", bbox_inches="tight")
plt.savefig("figures/png/cosmo_inference_c0_hod26.png", bbox_inches="tight")
plt.close()

ax = inference_plots.plot_corner(
    samples_list,
    hod_params,
    chain_labels,
    true_params=None,
    markers=[true_params],
)
plt.savefig("figures/pdf/hod_inference_c0_hod26.pdf", bbox_inches="tight")
plt.savefig("figures/png/hod_inference_c0_hod26.png", bbox_inches="tight")
