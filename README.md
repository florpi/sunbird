# SUNBIRD

SUNBIRD is a Python package that provides routines to train neural-network-based models for galaxy clustering. It also incorporates pre-trained models for different summary statistics, including:

- Galaxy two-point correlation function
- Density-split clustering statistics
- Void-galaxy cross-correlation function.

These models have been trained on mock galaxy catalogues based on the AbacusSummit simulations. The models are described in detail in Cuesta-Lazaro et al. (in preparation).

## Documentation

Documentation is hosted on Read the Docs, [pysunbird.readthedocs.io](https://pysunbird.readthedocs.io/).

## Requirements
Dependencies are listed in `pyproject.toml`, and installed automatically when installing the package.

Optional dependencies can be added for the inference routines, with the following command:
```bash
pip install sunbird[inference]
```


## Installation

### Install with pip
To install `sunbird`, you can use pip:
```bash
pip install sunbird @ git+https://github.com/florpi/sunbird.git
```

### Install from source
If you want to install the package from source, you can clone the repository and install it with:
```bash
git clone https://github.com/florpi/sunbird.git
cd sunbird
pip install .[inference]
```