# SUNBIRD

SUNBIRD is a Python package that provides routines to train neural-network-based models for galaxy clustering. It also incorporates pre-trained models for different summary statistics, including:

- Galaxy two-point correlation function
- Density-split clustering statistics
- Void-galaxy cross-correlation function.

These models have been trained on mock galaxy catalogues based on the AbacusSummit simulations. The models are described in detail in Cuesta-Lazaro et al. (in preparation).

## Documentation

Documentation is hosted on Read the Docs, [pysunbird.readthedocs.io](https://pysunbird.readthedocs.io/).

## Requirements

The following packages are required to run the code:

- black
- pytorch
- pandas
- numpy
- matplotlib
- xarray
- pytorch-lightning
- optuna
- joblib

## Installation

#### Cloning from repository

First
```
git clone https://github.com/florpi/sunbird.git
```
To install the code
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately)
```
python setup.py develop --user
```