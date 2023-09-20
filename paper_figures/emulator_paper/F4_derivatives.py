import jax
import argparse
import jax.numpy as jnp
import numpy as np
from sunbird.data.data_readers import Abacus 
from sunbird.summaries import Bundle
import matplotlib.pyplot as plt
from utils import get_names_labels
import scienceplots
import matplotlib
plt.style.use(['science', 'vibrant'])

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--loss', type=str, default='learned_gaussian')
    args.add_argument('--statistic', type=str, default='density_split_cross')
    args.add_argument('--hod_params', action=argparse.BooleanOptionalAction, default=False,)
    args = args.parse_args()

    loss = args.loss
    cosmology, hod_idx = 0, 26
    # Get parameters cosmo and hod
    abacus = Abacus(
        statistics=[args.statistic],
    )
    parameters = abacus.get_all_parameters(
        cosmology=cosmology,
    ).iloc[hod_idx].to_dict()
    # Get emulator
    emulator = Bundle(
        summaries=[args.statistic],
        loss=loss,
        flax=True,
    )
    s = emulator.coordinates['s']

    # Wrapper to compute 1D derivatives
    def wrapper(params, i, quantile=0, multipole=0,):
        emu_summary = emulator.all_summaries[args.statistic]
        inputs = emu_summary.input_transforms.transform(params)
        pred, var = emu_summary.model_apply(
            emu_summary.flax_params,
            inputs,
        )
        errors = jnp.sqrt(var)
        pred, var = emu_summary.apply_output_transforms(pred, errors)
        return pred[quantile][multipole][i]

    def compute_gradients(params, quantile, multipole):
        grad_i = jax.jit(jax.grad(wrapper))
        return jax.vmap(
            grad_i, 
            in_axes=(None, 0, None,None),
        )(params, jnp.arange(len(s)), quantile, multipole) # parallelize across s



    if args.hod_params:
        params_to_plot = ['logM1', 'B_cen', 'B_sat']
    else:
        params_to_plot = ['omega_cdm', 'sigma8_m', 'n_s',]
    labels = get_names_labels(params_to_plot)
    inputs = jnp.array([parameters[k] for k in emulator.input_names]).reshape(1, -1)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = cmap(np.linspace(0.01, 0.99, 5))
    fig, ax = plt.subplots(
        ncols = len(params_to_plot), 
        nrows=2, 
        figsize=(len(params_to_plot)*3.5, 2*2.6),
        sharex=True,
    )
    quintiles = [0,1,2,3]
    q_labels = [
        r'$\mathrm{Q}_0$', 
        r'$\mathrm{Q}_1$', 
        r'$\mathrm{Q}_3$', 
        r'$\mathrm{Q}_4$', 
    ]
    for i, param in enumerate(params_to_plot):
        param_idx = emulator.input_names.index(param)
        for q in quintiles:
            grad_values = compute_gradients(
                inputs,
                quantile=q,
                multipole=0,
            )
            grad_param_mono = grad_values[:len(s), 0, param_idx]
            grad_values = compute_gradients(
                inputs,
                quantile=q,
                multipole=1,
            )
            grad_param_quad = grad_values[:len(s), 0, param_idx]
            if q < 2:
                color = colors[quintiles[q]]
            else:
                color = colors[quintiles[q]+1]
            ax[0,i].plot(s, grad_param_mono, label=q_labels[q], color=color)
            ax[1,i].plot(s, grad_param_quad, label=q_labels[q], color=color)
        ax[1,i].set_xlabel(r'$s \, [h^{-1} \mathrm{Mpc}]$')
        if args.statistic == 'density_split_auto':
            ax[0,i].set_ylabel(rf'$\partial \xi^{{QQ}}_0 / \partial {labels[i]}$')
            ax[1,i].set_ylabel(rf'$\partial \xi^{{QQ}}_2 / \partial {labels[i]}$')
        else:
            ax[0,i].set_ylabel(rf'$\partial \xi^{{QG}}_0 / \partial {labels[i]}$')
            ax[1,i].set_ylabel(rf'$\partial \xi^{{QG}}_2 / \partial {labels[i]}$')

    ax[0,0].legend()
    # derivative correlation function respect to param 
    plt.tight_layout()
    if args.hod_params:
        plt.savefig(f'figures/png/F4_derivatives_{args.statistic}_hod.png',dpi=300)
        plt.savefig(f'figures/pdf/F4_derivatives_{args.statistic}_hod.pdf',dpi=300)
    else:
        plt.savefig(f'figures/png/F4_derivatives_{args.statistic}.png',dpi=300)
        plt.savefig(f'figures/pdf/F4_derivatives_{args.statistic}.pdf',dpi=300)


