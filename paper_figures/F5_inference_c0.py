from pathlib import Path
import matplotlib.pyplot as plt
from utils_plot_inference import labels, get_true_params, get_chain, plot_samples

plt.style.use(['science'])

chain_path = Path('/n/home11/ccuestalazaro/sunbird/scripts/inference/chains/')
cosmology = 0
hod_idx = 940
suffix = 'reduced_params_new'
true_params = get_true_params(cosmology=cosmology, hod_idx=hod_idx,)
samples_ds = get_chain(
    chain_path=chain_path,
    cosmology=cosmology,
    hod_idx=hod_idx,
    suffix=suffix,
)
colors = ['lightseagreen', 'mediumorchid', 'mediumorchid', 'mediumorchid', 'salmon', 'royalblue', 'rosybrown']
params_to_plot = [
    'omega_cdm', 'sigma8_m', 'n_s', 
    'logM1', 'logM_cut', 'alpha', 
    'alpha_s','alpha_c', 'sigma',
    'kappa', 'B_cen', 'B_sat',
] 
plot_samples(
    [samples_ds],
    None, 
    params_to_plot, 
    colors, 
    labels,
    markers=[
        true_params,
    ],
    markers_colors=colors,
)
plt.savefig(f'figures/png/Figure5_cosmo0.png', dpi=600, bbox_inches='tight')
plt.savefig(f'figures/pdf/Figure5_cosmo0.pdf', dpi=600, bbox_inches='tight')
