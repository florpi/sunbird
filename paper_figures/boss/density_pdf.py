"""
Figure 1: Density PDF
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
plt.style.use(['stylelib/science.mplstyle', 'stylelib/bright.mplstyle'])

data_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_density/CMASS/')
data_fn = data_dir / f'density_CMASS_NGC_zmin0.45_zmax0.6_zsplit_gaussian_NQ5_Rs10.npy'

data = np.load(data_fn, allow_pickle=True).item()

density = data['density']
quantiles_idx = data['quantiles_idx']

fig, ax = plt.subplots(figsize=(4, 4))
cmap = matplotlib.cm.get_cmap('coolwarm')
colors = cmap(np.linspace(0.01, 0.99, 5))

hist, bin_edges, patches = ax.hist(density, bins=200, density=True, lw=3.0, color='grey')

imin = 0
for i in range(5):
    dmax = density[quantiles_idx == i].max()
    imax = np.digitize([dmax], bin_edges)[0] - 1
    for index in range(imin, imax):
        patches[index].set_facecolor(colors[i])
    imin = imax

    ax.plot(np.nan, np.nan, color=colors[i], label=rf'${{\rm Q}}_{i}$', lw=4.0)
ax.set_xlabel(r'$\Delta \left(R_s = 10\, h^{-1}{\rm Mpc}\right)$', fontsize=15)
ax.set_ylabel('PDF', fontsize=15)
ax.set_xlim(-1.3, 4.0)
ax.legend(handlelength=1.0)
plt.tight_layout()
plt.savefig('fig/pdf/dsc_density_pdf.pdf')