from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots


args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper",
)
args.add_argument(
    "--loss",
    type=str,
    default="learned_gaussian",
)
args = args.parse_args()

chain_dir = Path(args.chain_dir)
# best hod for each cosmology
best_hod = {0: 26, 1: 74, 3: 30, 4: 15}

cosmo = 0
hod = best_hod[cosmo]

chain_handles = [
    f"cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf-ab=1-vb=1-ete=1-se=1",
    f"cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1",
    f"cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1",
]
chain_labels = [
    "AbacusSummit 2PCF",
    "AbacusSummit Density-Split",
    "AbacusSummit 2PCF + Density-Split",
]

cosmo_params = [
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "nrun",
    "N_eff",
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
plt.savefig("figures/pdf/F5_cosmo_c0_hod26.pdf", bbox_inches="tight")
plt.savefig("figures/png/F5_cosmo_c0_hod26.png", bbox_inches="tight")
plt.close()

ax = inference_plots.plot_corner(
    samples_list,
    hod_params,
    chain_labels,
    true_params=None,
    markers=[true_params],
)
plt.savefig("figures/pdf/F5_hod_c0_hod26.pdf", bbox_inches="tight")
plt.savefig("figures/png/F5_hod_c0_hod26.png", bbox_inches="tight")

ax = inference_plots.plot_corner(
    samples_list,
    cosmo_params + hod_params,
    chain_labels,
    true_params=None,
    markers=[true_params],
)
plt.savefig("figures/pdf/F5_full_c0_hod26.pdf", bbox_inches="tight")
plt.savefig("figures/png/F5_full_c0_hod26.png", bbox_inches="tight")
