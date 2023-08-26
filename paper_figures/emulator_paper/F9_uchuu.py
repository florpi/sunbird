from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots
from sunbird.data.data_readers import Uchuu


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

chain_dir = Path(args.chain_dir)

chain_handles = [
    f'cos=0-h=26-o=Uchuu-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Uchuu-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
]

chain_labels = [
    "Uchuu 2PCF",
    "Uchuu Density-Split+2PCF",
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

true_params = Uchuu().get_parameters_for_observation()

samples_list = []
for chain_handle in chain_handles:
    samples_list.append(
        inference_plots.get_MCSamples(
            chain_dir / chain_handle / "results.csv",
        )
    )

g = inference_plots.plot_corner(
    samples_list,
    cosmo_params[:3],
    chain_labels,
    true_params=None,
    markers=[true_params],
    inches_per_param=13.5 / 7,
)
plt.savefig("figures/pdf/F9_cosmo_uchuu.pdf", bbox_inches="tight")
plt.savefig("figures/png/F9_cosmo_uchuu.png", bbox_inches="tight")
plt.close()

ax = inference_plots.plot_corner(
    samples_list,
    hod_params,
    chain_labels,
    true_params=None,
    markers=[true_params],
)

plt.savefig("figures/pdf/F9_hod_uchuu.pdf", bbox_inches="tight")
plt.savefig("figures/png/F9_hod_uchuu.png", bbox_inches="tight")
