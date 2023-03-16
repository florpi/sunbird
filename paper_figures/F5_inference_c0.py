from pathlib import Path
import matplotlib.pyplot as plt
from utils_plot_inference import labels, get_true_params, get_chain, plot_samples

plt.style.use(["science"])

chain_path = Path("/n/home11/ccuestalazaro/sunbird/scripts/inference/chains/")
cosmology = 0
hod_idx = 940
suffix = None 
true_params = get_true_params(
    cosmology=cosmology,
    hod_idx=hod_idx,
)
samples_ds = get_chain(
    chain_path=chain_path,
    cosmology=cosmology,
    hod_idx=hod_idx,
    suffix=suffix,
)
samples_tpcf = get_chain(
    chain_path=chain_path,
    cosmology=cosmology,
    hod_idx=hod_idx,
    suffix='tpcf',
)

colors = [
    "lightseagreen",
    "mediumorchid",
    "mediumorchid",
    "mediumorchid",
    "salmon",
    "royalblue",
    "rosybrown",
]
params_to_plot = [
    "omega_cdm",
    "sigma8_m",
    "n_s",
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
plot_samples(
    [samples_tpcf,samples_ds],
    None,
    params_to_plot,
    colors,
    [r'$\mathrm{TPCF}$',r'$\mathrm{DS}_\mathrm{auto+cross}$'],
    markers=[
        true_params,
    ],
    markers_colors=['lightgray'],
)
plt.savefig(f"figures/png/Figure5.1_cosmo0.png", dpi=600, bbox_inches="tight")
plt.savefig(f"figures/pdf/Figure5.1_cosmo0.pdf", dpi=600, bbox_inches="tight")

params_to_plot = [
    "omega_cdm",
    "sigma8_m",
    "n_s",
]
plot_samples(
    [samples_tpcf,samples_ds],
    None,
    params_to_plot,
    colors,
    [r'$\mathrm{TPCF}$',r'$\mathrm{DS}_\mathrm{auto+cross}$'],
    markers=[
        true_params,
    ],
    markers_colors=['lightgray'],
)
plt.savefig(f"figures/png/Figure5.2_cosmo0_cosmology.png", dpi=600, bbox_inches="tight")
plt.savefig(f"figures/pdf/Figure5.2_cosmo0_cosmology.pdf", dpi=600, bbox_inches="tight")
