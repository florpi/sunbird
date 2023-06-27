import numpy as np
from cosmoprimo.fiducial import AbacusSummit, AbacusSummit_params
from sunbird.cosmology.growth_rate import Growth

def test_Omega_m():
    growth = Growth(emulate=True)
    cosmologies = list(range(5)) + list(range(130,181, 5))
    true_Omega_ms, pred_Omega_ms = [], []
    for cosmo in cosmologies:
        original_abacus = AbacusSummit(cosmo)
        abacus_params = AbacusSummit_params(name=cosmo)
        pred_Om = growth.Omega_m(
            omega_cdm=abacus_params['omega_cdm'],
            omega_b=abacus_params['omega_b'],
            h=abacus_params['h'],
            omega_ncdm=abacus_params['omega_ncdm'],
            w0_fld=abacus_params['w0_fld'],
            wa_fld=abacus_params['wa_fld'],
            z=0.5,
        )
        true_Omega_ms.append(original_abacus.Omega_m(0.5))
        pred_Omega_ms.append(pred_Om)
    np.testing.assert_allclose(true_Omega_ms, pred_Omega_ms, rtol=0.01)



def test_growth_given_true():
    growth = Growth(emulate=True)
    cosmologies = list(range(5)) + list(range(130,181, 5))
    true_growth, pred_growth= [], []
    for cosmo in cosmologies:
        original_abacus = AbacusSummit(cosmo)
        abacus_params = AbacusSummit_params(name=cosmo)
        pred_g= growth.approximate_growth_rate(
            omega_cdm=abacus_params['omega_cdm'],
            omega_b=abacus_params['omega_b'],
            h=abacus_params['h'],
            omega_ncdm=abacus_params['omega_ncdm'],
            w0_fld=abacus_params['w0_fld'],
            wa_fld=abacus_params['wa_fld'],
            z=0.5,
        )
        true_growth.append(original_abacus.growth_rate(0.5))
        pred_growth.append(pred_g)
    np.testing.assert_allclose(true_growth, pred_growth, rtol=0.01)

def test__solve_eq_abacus_emulated():
    redshift = 0.4
    growth = Growth(emulate=True)
    cosmologies = list(range(5)) + list(range(130,181, 5))
    true_growths = np.array([AbacusSummit(name=cosmo).growth_rate(redshift) for cosmo in cosmologies])
    omega_bs = np.array([AbacusSummit_params(name=cosmo)['omega_b'] for cosmo in cosmologies])
    omega_cdms = np.array([AbacusSummit_params(name=cosmo)['omega_cdm'] for cosmo in cosmologies])
    n_ss= np.array([AbacusSummit_params(name=cosmo)['n_s'] for cosmo in cosmologies])
    N_urs = np.array([AbacusSummit_params(name=cosmo)['N_ur'] for cosmo in cosmologies])
    w0_flds = np.array([AbacusSummit_params(name=cosmo)['w0_fld'] for cosmo in cosmologies])
    wa_flds = np.array([AbacusSummit_params(name=cosmo)['wa_fld'] for cosmo in cosmologies])
    sigma8s = np.array([AbacusSummit(name=cosmo).sigma8_m for cosmo in cosmologies])
    pred_growths = growth.get_growth(
        omega_b=omega_bs,
        omega_cdm=omega_cdms,
        sigma8=sigma8s,
        n_s = n_ss,
        w0_fld = w0_flds,
        wa_fld = wa_flds,
        N_ur=N_urs,
        z=redshift,
    )
    # assert that the maximum error is less than 2%
    np.testing.assert_allclose(true_growths, pred_growths, rtol=0.02)
    #  assert mean error is less than 1%
    mean_error = np.mean((pred_growths - true_growths) / true_growths)
    assert mean_error < 0.01

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
            z=redshift,
        )
        true_growths.append(original_abacus.growth_rate(redshift))
        pred_growths.append(pred_growth)
    # assert that the maximum error is less than 1%
    np.testing.assert_allclose(true_growths, pred_growths, rtol=0.01)