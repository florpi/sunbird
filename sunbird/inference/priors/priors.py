import yaml
import os
import importlib

def load_prior(prior_file):
    with open(prior_file) as f:
        config = yaml.safe_load(f)
    distributions_module = importlib.import_module(config.pop("stats_module"))
    prior = {}
    for param, dist_param in config['priors'].items():
        dist = getattr(distributions_module, dist_param.pop("distribution"))
        prior[param] = dist(**dist_param)
    return prior

class Yuan23:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        self.prior = load_prior(os.path.join(dirname, 'yuan23.yaml'))

class AbacusSummit:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        self.prior = load_prior(os.path.join(dirname, 'abacus_summit.yaml'))