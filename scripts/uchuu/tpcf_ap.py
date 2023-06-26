import time
import numpy as np
import pandas as pd
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
import time
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def downsample(df, target_nd, ranked_by_mass=True,):
    n = int(target_nd * boxsize **3)
    if ranked_by_mass:
        return df.sort_values(by='mgal', ascending=False).iloc[:n]
    return df.sample(n=n)

def get_rsd_positions(target_nd=3.e-4, ranked_by_mass=True, return_rsd=True):
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
        x_rsd = x + vx / (hubble * az)
        y_rsd = y + vy / (hubble * az)
        z_rsd = z + vz / (hubble * az)
        x_rsd = x_rsd % boxsize
        y_rsd = y_rsd % boxsize
        z_rsd = z_rsd % boxsize
        return x, y, z, x_rsd, y_rsd, z_rsd
    else:
        return x, y, z


def get_distorted_positions(positions, q_perp, q_para, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the 
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    positions_ap[:, 0] /= factor_x
    positions_ap[:, 1] /= factor_y
    positions_ap[:, 2] /= factor_z
    return positions_ap

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = [boxsize/factor_x, boxsize/factor_y, boxsize/factor_z]
    return boxsize_ap

if __name__ == '__main__':
    setup_logging(level='WARNING')
    uchuu_data = Path('../../data/clustering/uchuu/')
    overwrite = True
    nthreads = 32 
    save_mock = True
    boxsize = 2000
    cellsize = 5.0
    redshift = 0.57
    split = 'z'
    target_nd = 3.5e-4
    filter_shape = 'Gaussian'
    smoothing_radius = 10
    redges = np.hstack(
        [np.arange(0, 5, 1),
        np.arange(7, 30, 3),
        np.arange(31, 155, 5)]
    )
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)
    nquantiles = 5

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)
    # Uchuu cosmology
    Omega_m = 0.3089
    # cosmology of the mock as the truth
    mock_cosmo = AbacusSummit(Omega_m=Omega_m)
    az = 1 / (1 + redshift)
    hubble = 100 * mock_cosmo.efunc(redshift)
    # calculate distortion parameters
    q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
    q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
    q = q_perp**(2/3) * q_para**(1/3)
    for ranking in ['random', 'ranked']:
        print(f'Ranking = {ranking}')
        x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(
            target_nd=target_nd,
            ranked_by_mass=ranking == 'ranked',
        )
        tpcf_los = []
        for los in ['x', 'y', 'z']:
            if split == 'z':
                xpos = x_rsd if los == 'x' else x
                ypos = y_rsd if los == 'y' else y
                zpos = z_rsd if los == 'z' else z
            else:
                xpos, ypos, zpos = x, y, z

            data_positions = np.c_[xpos, ypos, zpos]

            data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                        q_perp=q_perp, q_para=q_para)
            boxsize_ap = np.array(get_distorted_box(boxsize=boxsize, q_perp=q_perp, q_para=q_para,
                                                    los=los))
            boxcenter_ap = boxsize_ap / 2

            # GALAXY 2PCF
            start_time = time.time()
            result = TwoPointCorrelationFunction(
                'smu', edges=edges, data_positions1=data_positions_ap,
                engine='corrfunc', boxsize=boxsize_ap, nthreads=nthreads,
                compute_sepsavg=False, position_type='pos', los=los,
            )
            s, multipoles = result(ells=(0, 2, 4), return_sep=True)
            tpcf_los.append(multipoles)
            # print(f'2PCF took {time.time() - start_time} sec')

        tpcf_los = np.asarray(tpcf_los)

        cout = {
            's': s,
            'multipoles': tpcf_los
        }
        output_dir = uchuu_data / f'tpcf'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = output_dir / f'tpcf_multipoles_{ranking}.npy'
        np.save(output_fn, cout)
