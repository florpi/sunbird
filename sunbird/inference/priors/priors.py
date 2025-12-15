import os
import yaml
import importlib


def load_prior(prior_file, stats_module='scipy.stats'):
    with open(prior_file) as f:
        config = yaml.safe_load(f)
    priors = {}
    for param, config_param in config.items():
        prior_param = config_param['prior']
        if prior_param['distribution'] == 'uniform':
            if stats_module == 'scipy.stats':
                min_uniform = prior_param.pop('min')
                max_uniform = prior_param.pop('max')
                prior_param['loc'] = min_uniform
                prior_param['scale'] = max_uniform - min_uniform
            elif stats_module == 'numpyro.distributions':
                prior_param['low'] = prior_param.pop('min')
                prior_param['high'] = prior_param.pop('max')
                prior_param['distribution'] = 'Uniform'
        dist = getattr(importlib.import_module(stats_module),
                       prior_param.pop("distribution"))
        priors[param] = dist(**prior_param)
    return priors

def load_ranges(prior_file):
    with open(prior_file) as f:
        config = yaml.safe_load(f)
    ranges = {}
    for param, config_param in config.items():
        prior_param = config_param['prior']
        if prior_param['distribution'] == 'uniform':
            ranges[param] = [prior_param['min'], prior_param['max']]
    return ranges

def load_labels(prior_file):
    with open(prior_file) as f:
        config = yaml.safe_load(f)
    labels = {}
    for param, config_param in config.items():
        label_param = r'$' + config_param['latex'] + '$'
        labels[param] = label_param
    return labels


class Yuan23:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors = load_prior(os.path.join(dirname, 'yuan23.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'yuan23.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'yuan23.yaml'))

class Rocher25:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors = load_prior(os.path.join(dirname, 'rocher25.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'rocher25.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'rocher25.yaml'))

class DESIEDR:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors = load_prior(os.path.join(dirname, 'desi_edr.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'desi_edr.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'desi_edr.yaml'))

class AbacusSummit:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors =  load_prior(os.path.join(dirname, 'abacus_summit.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'abacus_summit.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'abacus_summit.yaml'))

class AbacusPNG:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors =  load_prior(os.path.join(dirname, 'abacus_png.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'abacus_png.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'abacus_png.yaml'))

class AbacusGrowthMC:
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors =  load_prior(os.path.join(dirname, 'abacus_mc.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'abacus_mc.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'abacus_mc.yaml'))

# class PocoMCAbacusSummitPrior:
#     def __init__(self, stats_module='scipy.stats'):
#         from .abacus_summit import AbacusSummitEllipsoid
#         dirname = os.path.dirname(__file__)
#         self.priors =  load_prior(os.path.join(dirname, 'abacus_summit.yaml'),
#                           stats_module=stats_module)
#         self.ranges = load_ranges(os.path.join(dirname, 'abacus_summit.yaml'))
#         self.labels = load_labels(os.path.join(dirname, 'abacus_summit.yaml'))

#         self.ellipsoid = AbacusSummitEllipsoid(params=self.priors.keys())

#     def logpdf(self, x):
#         if self.ellipsoid.is_within(x):
#             return 0.0
#         return -np.inf

#     def rvs(self, size=1)
#         return np.array([self.priors[param].rvs(size=size) for param in self.priors.keys()]).T

