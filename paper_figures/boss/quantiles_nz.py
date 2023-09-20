"""
Figure 2: n(z) distribution of quantiles and galaxies
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmoprimo.fiducial import AbacusSummit
import fitsio
import healpy as hp

plt.style.use(['stylelib/science.mplstyle', 'stylelib/bright.mplstyle'])

def spl_nofz(zarray, fsky, cosmo, zmin, zmax, Nzbins=100):
    zbins = np.linspace(zmin, zmax, Nzbins+1)
    Nz, zbins = np.histogram(zarray, zbins)

    zmid = zbins[0:-1] + (zmax-zmin)/Nzbins/2.0
    # set z range boundaries to be zmin and zmax and avoid the interpolation error
    zmid[0], zmid[-1] = zbins[0], zbins[-1]

    rmin = cosmo.comoving_radial_distance(zbins[0:-1])
    rmax = cosmo.comoving_radial_distance(zbins[1:])

    vol = fsky * 4./3*np.pi * (rmax**3.0 - rmin**3.0)
    nz_array = Nz/vol
    spl_nz = InterpolatedUnivariateSpline(zmid, nz_array)
    return spl_nz

def sky_fraction(randoms_fn):
    """Compute the sky fraction of a randoms catalogue."""
    data = np.genfromtxt(randoms_fn)
    ra = data[:, 0]
    dec = data[:, 1]
    nside = 512
    npix = hp.nside2npix(nside)
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    pixel_indices = hp.ang2pix(nside, theta, phi)
    pixel_unique, counts = np.unique(pixel_indices, return_counts=True)
    fsky = len(pixel_unique)/npix
    return fsky

fsky = 0.166
cosmo = AbacusSummit(0)
zmin, zmax = 0.45, 0.6


data_dir = '/pscratch/sd/e/epaillas/ds_boss/ds_quantiles/CMASS/'
data_fn = Path(data_dir) / f'ds_quantiles_CMASS_NGC_zmin0.45_zmax0.6_zsplit_gaussian_NQ5_Rs10_default_FKP_padded.npy'
quantiles_positions = np.load(data_fn, allow_pickle=True)

fig, ax = plt.subplots(figsize=(5, 4))
cmap = matplotlib.cm.get_cmap('coolwarm')
colors = cmap(np.linspace(0.01, 0.99, 5))

data_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/CMASS/')
data_fn = data_dir / f'galaxy_DR12v5_CMASS_North.fits.gz'
data = fitsio.read(data_fn)
mask = (data['Z'] > zmin) & (data['Z'] < zmax)
z = data[mask]['Z']
nz = spl_nofz(z, fsky, cosmo, zmin, zmax)
ax.scatter(z, nz(z)*1e4, s=1.0, c='#404040',)

for ds in [0, 1, 3, 4]:
    quintile = np.array(quantiles_positions[ds], dtype=float)
    z = quintile[:, 2]
    nz = spl_nofz(z, fsky, cosmo, zmin, zmax)
    ax.scatter(z, nz(z)*1e4, s=1.0, color=colors[ds])
    ax.plot(np.nan, np.nan, color=colors[ds], label=rf'${{\rm Q}}_{ds}$')

ax.plot(np.nan, np.nan, color='#404040', label='galaxies',)
    
ax.set_xlabel('redshift 'r'$z$', fontsize=15)
ax.set_ylabel(r'$n(z)\,[10^4 h^{3}{\rm Mpc^{-3}}]$', fontsize=15)
ax.legend(fontsize=15, handlelength=1.0, loc='lower center')
plt.tight_layout()
plt.savefig('fig/png/quintiles_nz_cmass.png', dpi=300)