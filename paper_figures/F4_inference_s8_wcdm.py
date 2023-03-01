from pathlib import Path
import matplotlib.pyplot as plt
from utils_plot_inference import labels, get_true_params, get_chain, plot_samples

plt.style.use(['science'])

chain_path = Path('/n/home11/ccuestalazaro/sunbird/scripts/inference/chains/')
hod_idx = {
    0: 940,
    1: 491,
    3: 441, 
    4: 719, 
}
suffix = 'reduced_params' 

true_params, samples_ds = [], []
for cosmology in hod_idx.keys():
    true_params.append(
        get_true_params(cosmology, hod_idx[cosmology],)
    )
    samples_ds.append(
        get_chain(
            chain_path=chain_path,
            cosmology=cosmology,
            hod_idx= hod_idx[cosmology],
            suffix=suffix,
        )
    )    


colors = ['lightseagreen', 'mediumorchid', 'mediumorchid', 'mediumorchid', 'salmon', 'royalblue', 'rosybrown']
params_to_plot = ['omega_cdm', 'sigma8_m', 'n_s'] 


labels = [
    r'Low $\omega_{\rm cdm}$',
    r'High $\omega_{\rm cdm}$',
]
plot_samples(
    [samples_ds[1], samples_ds[2]], 
    None, 
    params_to_plot, 
    colors, 
    labels,
    markers=[
        true_params[1],
        true_params[2],
    ],
    markers_colors=colors,
)
plt.savefig(f'figures/png/Figure4.2_infer_wcdm_{suffix}.png', dpi=600, bbox_inches='tight')
plt.savefig(f'figures/pdf/Figure4.2_infer_wcdm_{suffix}.pdf', dpi=600, bbox_inches='tight')


labels = [
    r'High $\sigma_8$',
    r'Low $\sigma_8$',
]
plot_samples(
    [samples_ds[0], samples_ds[3]], 
    None, 
    params_to_plot, 
    colors, 
    labels,
    markers=[
        true_params[0],
        true_params[3],
    ],
    markers_colors=colors,
)
plt.savefig(f'figures/png/Figure4.2_infer_s8_{suffix}.png', dpi=600, bbox_inches='tight')
plt.savefig(f'figures/pdf/Figure4.2_infer_s8_{suffix}.pdf', dpi=600, bbox_inches='tight')