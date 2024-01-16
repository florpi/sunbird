===============
Getting started
===============

Loading pre-trained models
==========================

SUNBIRD comes with pre-trained emulators for different summary statistics, which were calibrated to
work for specific samples of galaxies. These include

- Two-point correlation function emulator calibrated for the BOSS DR12 CMASS sample.
- Density-split clustering emulator calibrated for the BOSS DR12 CMASS sample.
- Density-split clustering emulator calibrated for the Beyond2PT challenge LCDM mocks.
- Void-galaxy cross-correlation emulator calibrated for the BOSS DR12 CMASS sample.

To get an emulator prediction, you can simply call the corresponding class with the desired
parameters. For example, to get the TPCF prediction for the BOSS DR12 CMASS sample, you can do::

    from sunbird.summaries import TPCF

    emulator = TPCF()
    s = emulator.coordinates['s']

    select_filters = {
        'multipoles': [0]
    }
    slice_filters = {
        's': [1, 150]
    }

    cosmo_params = {'omega_b': 0.02, 'omega_cdm': 0.12, 'sigma8_m': 0.8,
                    'n_s': 0.96, 'nrun': 0.0, 'N_ur':2.03, 'w0_fld': -1.0, 'wa_fld': 0.0}

    hod_params = {'logM1': 13.9, 'logM_cut': 12.6, 'alpha': 0.75, 'alpha_s': 1.0,
                'alpha_c': 0.3, 'logsigma': -1.8, 'kappa': 0.1, 'B_cen': 0.0, 'B_sat': 0.0}

    parameters = {**cosmo_params, **hod_params}

    prediction = emulator(
        param_dict=parameters,
        select_filters=select_filters,
        slice_filters=slice_filters,
    ).numpy()   
