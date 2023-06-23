import pandas as pd
import numpy as np
from pathlib import Path
from pycorr import setup_logging, TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.cosmology import Cosmology
from scipy.interpolate import RectBivariateSpline
from scipy import special
import time
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def downsample(df, target_nd, ranked_by_mass=True,):
    n = int(target_nd * boxsize **3)
    if ranked_by_mass:
        return df.sort_values(by='mgal', ascending=False).iloc[:n]
    return df.sample(n=n)


def read_mock(target_nd=3.e-4, ranked_by_mass=True, return_rsd=True):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    df = pd.read_csv(
        '../../data/uchuu_mocks/mpeak_scat0.1_n3.5E4.dat',
        names=['mgal', 'mpeak', 'id', 'upid', 'x', 'y', 'z', 'vx', 'vy', 'vz'], 
        sep= ' ',
    )
    df = downsample(df, target_nd=target_nd, ranked_by_mass=ranked_by_mass)
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    vx = df['vx'].values
    vy = df['vy'].values
    vz = df['vz'].values
    if return_rsd:
        x_rsd = x + vx / (Hz * az)
        y_rsd = y + vy / (Hz * az)
        z_rsd = z + vz / (Hz * az)
        x_rsd = x_rsd % boxsize
        y_rsd = y_rsd % boxsize
        z_rsd = z_rsd % boxsize
        return x, y, z, x_rsd, y_rsd, z_rsd
    else:
        return x, y, z

def project_to_multipoles(r_c, mu_edges, corr, ells=(0,2,4)):
    """Return the multiple decomposition of an input correlation
    function measurement"""
    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(ells)
    toret = []
    for ill,ell in enumerate(ells):
        dmu = np.diff(mu_edges, axis=-1)
        poly = special.legendre(ell)(mu_edges)
        legendre = (2*ell + 1) * (poly[1:] + poly[:-1])/2. * dmu
        toret.append(np.sum(corr*legendre, axis=-1)/np.sum(dmu))
    return r_c, toret

def get_distorted_multipoles(result, q_perp, q_para, ells=(0, 2)):
    """Given an input correlation function measured by pycorr,
    return the multipoles distorted by the Alcock-Paczynski effect"""
    s = result.sep[:, 0]
    mu = result.seps[1][0, :]
    S, MU = np.meshgrid(s, mu)
    true_sperp = (S * np.sqrt(1 - MU ** 2) * q_perp).T
    true_spara = (S * MU * q_para).T

    true_s = np.sqrt(true_sperp**2 + true_spara**2)
    true_mu = true_spara / true_s

    xi_func = RectBivariateSpline(s, mu, result.corr, kx=1, ky=1)
    true_xi = np.zeros_like(result.corr)
    for i in range(len(s)):
        for j in range(len(mu)):
            true_xi[i, j] = xi_func(true_s[i, j], true_mu[i, j])

    s, multipoles = project_to_multipoles(r_c=s, mu_edges=np.linspace(-1, 1, 241),
                                          corr=true_xi, ells=ells)
    return s, multipoles


if __name__ == '__main__':
    setup_logging(level='WARNING')
    uchuu_data = Path('../../data/clustering/uchuu/')
    boxsize = 2000
    redshift = 0.57
    # redges = np.hstack([np.arange(0, 5, 1), np.arange(7, 30, 3), np.arange(30, 155, 5)])
    redges = np.hstack([np.arange(0, 5, 1), np.arange(7, 30, 3), np.arange(31, 155, 5)]) # fixed spacing
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    # Patchy cosmology as our fiducial
    fid_cosmo = Cosmology(
        Omega_m=0.307115,
        Omega_b=0.048,
        sigma8=0.8288,
        h=0.677,
        engine='class'
    )
    # Uchuu cosmology
    Omega_m = 0.3089
    Omega_l = 1 - Omega_m
    H_0 = 100.0
    az = 1 / (1 + redshift)
    Hz = H_0 * np.sqrt(Omega_m * (1 + redshift)**3 + Omega_l)
    mock_cosmo = AbacusSummit(Omega_m=Omega_m)
    # calculate distortion parameters
    q_perp = mock_cosmo.comoving_angular_distance(0.5)/fid_cosmo.comoving_angular_distance(0.5)
    q_para = fid_cosmo.hubble_function(0.5)/mock_cosmo.hubble_function(0.5)
    q = q_perp**(2/3) * q_para**(1/3)
    print(f'q_perp = {q_perp:.3f}')
    print(f'q_para = {q_para:.3f}')
    print(f'q = {q:.3f}')

    for ranking in ['ranked', 'random']:
        start_time = time.time()
        x, y, z, x_rsd, y_rsd, z_rsd = read_mock(ranked_by_mass=True if ranking=='ranked' else False)

        auto_los = []
        for los in ['x', 'y', 'z']:
            xpos = x_rsd if los == 'x' else x
            ypos = y_rsd if los == 'y' else y
            zpos = z_rsd if los == 'z' else z

            data_positions = np.c_[xpos, ypos, zpos]

            # AUTOCORRELATION FUNCTION
            result = TwoPointCorrelationFunction(
                'smu', edges=edges, data_positions1=data_positions,
                los=los, engine='corrfunc', boxsize=boxsize, nthreads=16,
                compute_sepsavg=False, position_type='pos'
            )
            s, multipoles = get_distorted_multipoles(result, q_perp, q_para, ells=(0, 2, 4))
            auto_los.append(multipoles)

        auto_los = np.asarray(auto_los)

        cout = {
            's': s,
            'multipoles': auto_los
        }
        output_dir =  uchuu_data / 'xi_smu/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = Path(output_dir, f'xi_smu_{ranking}.npy')
        np.save(output_fn, cout)

        end_time = time.time() - start_time
        print(f'Uchuu took {end_time:.3f} sec')