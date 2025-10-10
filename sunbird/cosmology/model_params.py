

def get_model_params(cosmo_model='base', hod_model='base'):
    """
    Returns a list with the free parameters of the model for an
    input model string. 

    Examples: 'base' for base-LCDM. 'base_w0wa' to add w0wa dark
    energy parameters. 
    """
    params = []
    # cosmology
    if 'base' in cosmo_model:
        params += ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    if 'w0' in cosmo_model:
        params += ['w0_fld']
    if 'wa' in cosmo_model:
        params += ['wa_fld']
    if 'Nur' in cosmo_model:
        params += ['N_ur']
    if 'nrun' in cosmo_model:
        params += ['nrun']
    # HOD
    if 'base' in hod_model:
        params += ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa']
    if 'AB' in hod_model:
        params += ['A_cen', 'A_sat', 'B_cen', 'B_sat']
    if 'VB' in hod_model:
        params += ['alpha_c', 'alpha_s']
    if 's' in hod_model:
        params += ['s']
    return params