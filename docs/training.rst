======================
Training new emulators
======================

.. warning:: 
    This section is still under construction. Please check back later for updates.

SUNBIRD implements routines to train neural-network emulators with pytorch. The first step for doing so
is defining a training set of input-output pairs. Examples of training sets include:

- Galaxy power spectra measured from mock catalogues with different cosmologies and HOD parameters.
- Marked power spectrua measured from mock catalogues with a fixed cosmology, but different HOD parameters.

We will work out an example where we implement a new emulator for the void-galaxy cross-correlation function,
which will capture the dependence of this statistic on cosmology and HOD parameters.

Let's assume that we have measured the void-galaxy cross-correlation function multipoles (monopole,
quadrupole and hexadecapole) using 50 radial bins, from the AbacusSummit simulations for 85 different
cosmologies, each cosmology having 100 different HOD parameters. For each cosmology, we will compress
the measurements into a single ``numpy`` array of shape ``(100, 3, 50)``,
and save the output in a dictionary with the following structure::

    import numpy as np

    for cosmo in range(85):

        # s is the array of radial bins, with shape (50,)
        # multipoles is the array of multipoles, with shape (100, 3, 50)
        # cosmo is the cosmology index (0 <= cosmo < 85)

        output = {'s': s, 'multipoles': multipoles}
        np.save(f'void_multipoles_c{cosmo}.npy', output)

Once we have generated the compressed measurements for all the cosmologies, we will place them
under the ``data/clustering/abacus/{dataset}/voids`` directory. Here, ``dataset`` is a name that is
used to identify the training set (you might want to train a different emulator at some point, perhaps
with a different set of cosmologies and/or HOD parameters, so it is useful to have a name that
identifies the training set). In our example, we will use ``dataset = 'tutorial'``.

.. note::

    The ``data/`` directory (and its associated subdirectories) is not tracked by ``git``,
    so you will have to create it manually, unless you have downloaded the ``data.tar.gz``
    file from Globus.

Alongside the clustering measurements, we will also need to provide a file with the cosmological
parameters and HOD parameters for each of the cosmologies. Each cosmology should have its own
CSV file placed under ``data/parameters/abacus/{dataset}``, where each row will contain the
cosmological parameters and HOD parameters for each HOD variation. An example of such a file
would be

.. list-table:: AbacusSummit_c000.csv
    :widths: 15 15 15 15
    :header-rows: 1

    * - omega_b
      - omega_cdm
      - sigma8_m
      - n_s
    * - 0.0
      - 0.0
      - 0.0
      - 0.0
    * - 0.1
      - 0.0
      - 0.0
      - 0.0
    * - ...
      - ...
      - ...
      - ...
