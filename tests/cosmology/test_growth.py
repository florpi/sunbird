import numpy as np
from cosmoprimo.fiducial import AbacusSummit, AbacusSummit_params
from sunbird.cosmology.growth_rate import Growth

def test__solve_eq_abacus():
    redshift = 0.4
    growth = Growth()
    cosmologies = list(range(5)) + list(range(130,181, 5))
    true_growths, pred_growths = [], []
    for cosmo in cosmologies:
        original_abacus = AbacusSummit(cosmo)
        abacus_params = AbacusSummit_params(name=cosmo)
        pred_growth = growth.get_growth(
            omega_b = abacus_params['omega_b'],
            omega_cdm = abacus_params['omega_cdm'],
            sigma8 = original_abacus.sigma8_m,
            n_s = abacus_params['n_s'],
            N_ur = abacus_params['N_ur'],
            w0_fld = abacus_params['w0_fld'],
            wa_fld = abacus_params['wa_fld'],
            redshift=redshift,
        )
        true_growths.append(original_abacus.growth_rate(redshift))
        pred_growths.append(pred_growth)
    # assert that the maximum error is less than 1%
    np.testing.assert_allclose(true_growths, pred_growths, rtol=0.01)


