import os
import yaml
import importlib

class PriorBase:
    """
    Base class for loading priors from a YAML configuration file.
    """
    def __init__(self, filename: str, stats_module: str = 'scipy.stats', add_latex: bool = True):
        """        
        Parameters
        ----------
        filename : str
            Path to the YAML configuration file containing the priors.
        stats_module : str, optional
            The statistics module to use for the priors. Defaults to 'scipy.stats'
        add_latex : bool, optional
            Whether to add LaTeX formatting to the labels. Defaults to True
        """
        self.filename = filename 
        self.priors = self.load_prior(filename, stats_module=stats_module)
        self.ranges = self.load_ranges(filename)
        self.labels = self.load_labels(filename, add_latex=add_latex)

    @staticmethod
    def load_prior(filename: str, stats_module: str = 'scipy.stats') -> dict:
        """
        Load priors from a YAML configuration file.

        Parameters
        ----------
        filename : str
            Path to the YAML configuration file containing the priors.
        stats_module : str, optional
            The statistics module to use for the priors. Defaults to 'scipy.stats'

        Returns
        -------
        dict
            A dictionary where keys are parameter names and values are prior distributions.
        """
        with open(filename) as f:
            config = yaml.safe_load(f)
        priors = {}
        for param_name, param_config in config.items():
            prior = param_config['prior']
            if prior['distribution'] == 'uniform':
                if stats_module == 'scipy.stats':
                    min_uniform = prior.pop('min')
                    max_uniform = prior.pop('max')
                    prior['loc'] = min_uniform
                    prior['scale'] = max_uniform - min_uniform
                elif stats_module == 'numpyro.distributions':
                    prior['low'] = prior.pop('min')
                    prior['high'] = prior.pop('max')
                    prior['distribution'] = 'Uniform'
            dist = getattr(importlib.import_module(stats_module), prior.pop("distribution"))
            priors[param_name] = dist(**prior)
        return priors
    
    @staticmethod
    def load_ranges(filename: str) -> dict[str, list]:
        """
        Load parameter ranges from a YAML configuration file.
        This includes only 'uniform' prior distributions (min, max).
        
        Parameters
        ----------
        filename : str
            Path to the YAML configuration file containing the priors.

        Returns
        -------
        dict[str, list]
            A dictionary where keys are parameter names and values are parameter ranges.
        """
        with open(filename) as f:
            config = yaml.safe_load(f)
        ranges = {}
        for param_name, param_config in config.items():
            prior = param_config['prior']
            if prior['distribution'] == 'uniform':
                ranges[param_name] = [prior['min'], prior['max']]
        return ranges
    
    @staticmethod
    def load_labels(filename: str, add_latex: bool = True) -> dict[str, str]:
        """
        Load parameter labels from a YAML configuration file.

        Parameters
        ----------
        filename : str
            Path to the YAML configuration file containing the priors.
        add_latex : bool, optional
            Whether to add LaTeX formatting to the labels. Defaults to True

        Returns
        -------
        dict[str, str]
            A dictionary where keys are parameter names and values are parameter labels.
        """
        with open(filename) as f:
            config = yaml.safe_load(f)
        labels = {}
        for param_name, param_config in config.items():
            label = param_config['latex']
            if add_latex:
                label = r'$' + label + '$'
            labels[param_name] = label
        return labels

class Yuan23(PriorBase):
    """
    Prior class for LRG HOD parameters based on Yuan et al. 2023.
    """
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'yuan23.yaml')
        super().__init__(filename, stats_module=stats_module)

class Rocher25(PriorBase):
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'rocher25.yaml')
        super().__init__(filename, stats_module=stats_module)
        
class Bouchard25(PriorBase):
    """
    Prior class for BGS HOD parameters defined by S.Bouchard in 2025.
    """
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'bouchard25.yaml')
        super().__init__(filename, stats_module=stats_module)

class DESIEDR(PriorBase):
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'desi_edr.yaml')
        super().__init__(filename, stats_module=stats_module)
        
class AbacusSummit(PriorBase):
    """
    Prior class for AbacusSummit cosmological parameters.
    """
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'abacus_summit.yaml')
        super().__init__(filename, stats_module=stats_module)

class AbacusPNG(PriorBase):
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'abacus_png.yaml')
        super().__init__(filename, stats_module=stats_module)

class AbacusGrowthMC(PriorBase):
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'abacus_mc.yaml')
        super().__init__(filename, stats_module=stats_module)


def get_priors(
    cosmo_class: PriorBase = AbacusSummit, 
    hod_class: PriorBase = Yuan23,
    cosmo: bool = True, 
    hod: bool = True,
    stats_module: str = 'scipy.stats',
) -> tuple[dict, dict, dict]:
    """
    Get the priors, ranges and labels for the parameters.

    Parameters
    ----------
    cosmo_class : PriorBase, optional
        Class for the cosmological parameters. Defaults to AbacusSummit.
    hod_class : PriorBase, optional
        Class for the HOD parameters. Defaults to Yuan23.
    cosmo : bool, optional
        Whether to include the cosmological parameters. Defaults to True.
    hod : bool, optional
        Whether to include the HOD parameters. Defaults to True.

    Returns
    -------
    priors : dict
        Dictionary containing the priors for the parameters.
    ranges : dict
        Dictionary containing the prior ranges for the parameters.
    labels : dict
        Dictionary containing the labels for the parameters.
    """
    priors, ranges, labels = {}, {}, {}  
    if cosmo:
        priors.update(cosmo_class(stats_module).priors)
        ranges.update(cosmo_class(stats_module).ranges)
        labels.update(cosmo_class(stats_module).labels)
    if hod:
        priors.update(hod_class(stats_module).priors)
        ranges.update(hod_class(stats_module).ranges)
        labels.update(hod_class(stats_module).labels)
    return priors, ranges, labels

def in_range(values: dict, range: dict, verbose: bool = False) -> bool:
    """
    Check if the values are within the given range.

    Parameters
    ----------
    values : dict
        Dictionary containing the parameter values.
    range : dict
        Dictionary containing the parameter ranges.
    verbose : bool, optional
        Whether to print the parameters that are not within the range. Defaults to False.

    Returns
    -------
    bool
        True if all values are within the range, False otherwise.
    """
    all_within = True
    for param, val in values.items():
        if param in range:
            min_val, max_val = range[param]
            if not (min_val <= val <= max_val):
                all_within = False
                if verbose:
                    print(f'Parameter {param} with value {val} is out of range [{min_val}, {max_val}]')
    return all_within




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

