from scipy import optimize
import numpy as np
import pandas.DataFrame as pd
from scipy.stats import uniform
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.fiducial import DESI

class Growth:
    def __init__(self, theta_star: float=AbacusSummit(0).theta_star, emulate=False,):
        self.theta_star = theta_star
        self.emulate = emulate
        if self.emulate:
            self.growth_emulator = self.load_emulator()
        
    def generate_emulator_training_data(self, n_samples: int = 10,):
        parameters = {
            'omega_b': [0.0207, 0.0243],
            'omega_cdm': [0.1032, 0.140],
            'sigma8': [0.678, 0.938],
            'N_ur': [1.188, 2.889],
            'n_s': [0.9012, 1.025],
            'w0_fld': [-1.22, -0.726],
            'wa_fld': [-0.628, 0.621],
        }
        samples = [np.random.uniform(low, high, n_samples) for low, high in parameters.values()]
        samples_matrix = np.stack(samples, axis=-1)
        print(samples_matrix.shape)
        h_values = []
        for sample in samples_matrix:
            cosmology = self.get_cosmology_fixed_theta_star(
                DESI(engine='class'), 
                dict(
                    theta_star=self.theta_star,
                    omega_b = sample[0],
                    omega_cdm = sample[1],
                    sigma8 = sample[2],
                    N_ur=sample[3],
                    n_s = sample[4],
                    w0_fld = sample[5],
                    wa_fld = sample[5],
                ),
            )
            h_values.append(cosmology.h)
        h_values = np.array(h_values)
        data_dict = {param: samples[:,i] for i, param in enumerate(parameters.keys())}
        data_dict['h'] = h_values
        return pd.DataFrame(
            data_dict
        )

    def load_emulator():
        pass

    def get_cosmology_fixed_theta_star(self, fiducial, params, h_limits=[0.5,0.9], xtol=1.e-6,):
        theta = params.pop('theta_star', None)
        fiducial = fiducial.clone(base='input', **params)
        if theta is not None:
            if 'h' in params:
                raise ValueError('Cannot provide both theta_star and h')
            def f(h):
                cosmo = fiducial.clone(base='input', h=h)
                return 100. * (theta - cosmo.get_thermodynamics().theta_star)

            rtol = xtol
            try:
                h = optimize.bisect(f, *h_limits, xtol=xtol, rtol=rtol, disp=True)
            except ValueError as exc:
                raise ValueError('Could not find proper h value in the interval that matches theta_star = {:.4f} with [f({:.3f}), f({:.3f})] = [{:.4f}, {:.4f}]'.format(theta, *limits, *list(map(f, limits)))) from exc
            cosmo = fiducial.clone(base='input', h=h)
        return cosmo

    def get_growth(
        self, 
        omega_b: float,
        omega_cdm: float,
        sigma8: float,
        n_s: float,
        N_ur: float, 
        w0_fld: float,
        wa_fld: float,
        redshift: float,
        
    ):
        if self.emulate:
            # what exactly should be input?
            h = self.growth_emulator(
                omega_b=omega_b,
                omega_cdm=omega_cdm,
                sigma8=sigma8,
                N_ur=N_ur,
                n_s=n_s,
            )
            cosmology = DESI(
                dict(
                    omega_b = omega_b,
                    omega_cdm = omega_cdm,
                    sigma8 = sigma8,
                    n_s = n_s,
                    N_ur=N_ur,
                    w0_fld = w0_fld,
                    wa_fld = wa_fld,
                    h=h,
                ),
            )
        else:
            cosmology = self.get_cosmology_fixed_theta_star(
                DESI(engine='class'), 
                dict(
                    theta_star=self.theta_star,
                    omega_b = omega_b,
                    omega_cdm = omega_cdm,
                    sigma8 = sigma8,
                    n_s = n_s,
                    N_ur=N_ur,
                    w0_fld = w0_fld,
                    wa_fld = wa_fld,
                ),
            )
        return cosmology.growth_rate(redshift)

if __name__ == '__main__':
    growth = Growth()
    data = growth.generate_emulator_training_data()
    print(data)