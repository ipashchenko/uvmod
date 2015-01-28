uvmod
=====

**Simple model in uv-plane**

``uvmod`` - tool used to fit simple models to radiointerferomatric data designed for
Radioastron data processing. It currently fits apmlitudes only but is supposed to be extended.

Documentation
-------------

Requirements:
^^^^^^^^^^^^^
``numpy``, ``scipy`` (for LMA), ``emcee`` (for MCMC), ``pylab`` & ``triangle_plot`` (for plots)

Using  Levenberg–Marquardt algorithm (LMA) with ``-leastsq`` key:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``user@host:~$ python fit_amp.py detections.dat -leastsq -savefile file -savefig fig.png -p0 p0_0 p0_1 [ p0_2 ... ]``

Parameters:

- ``-p0`` - starting estimates for the minimization. Minimum 2 parameters (1D case).

Notes:

- Can't handle limits (nondetections).

- You should know all requirements and weakness of LMA and LSQ in general.

Fitting detections and upper limits with gaussian model and gaussian noise:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``user@host:~$ python fit_amp.py detections.dat ulimits.dat -savefile file -savefig fig.png -max_p max_p0 max_p1 [ max_p2 ... ]  [ -p0 p0_0 p0_1 [ p0_2 ... ]] [ -std0 std0_0 std_1 [ std0_2 ... ]]``

Parameters:

- ``-max_p`` - will use uniform prior distributions on parameters with ranges ``[0, max_p_i]`` for parameter ``p_i``. If you are using model with ``n`` parameters then specify ``n`` values. Minimum 2 parameters (1D case).

- ``-p0`` - center of ball with initial parameter values (used for MCMC initialization). Minimum 2 parameters (1D case).

- ``-std0`` - stds of gaussains with centers in ``p0`` (used for MCMC initialization). Minimum 2 parameters (1D case).

Notes:

- 1D model provides 2 parameters (amplitude and std of gaussian). If uncertainties are unknown then third parameter will be normal noise dispersion.

- If no ``-p0`` & ``-std0`` options are used then initialize MCMC in n-dimensional cube [0, 1]**N.

- If no ``-std0`` option is used then radius (std) of MCMC initialization is 0.01. 

- If uncertainties of data are known one can introduce additional variance - *jitter*. It is last parameter in parameters list.

- One can also model *outliers* in data. That requires additional 3 parameters: amplitude, mean and variance of outliers distribution. It is not implemented for data with limits.

- Only uniform priors are implemented (``[0, max_p]``).

- Most of modelling used receipts in `Hogg's et al. paper`_.

.. _Hogg's et al. paper: http://arxiv.org/abs/1008.4686

Quering data for source from RA DB:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``user@host:~$ python query_source.py -source source_name -band band_name [ -baselines bl_from bl_to ] [ -ra ] -savefig fig.png -savefile data.dat -user user -password password``

Parameters:

- ``-source`` - source name (B1950 or it's name in RA DB).

- ``-band`` - frequency band: p, l, c or k.

- ``-pols`` - polarizations: rr, ll, rl, lr. If option isn't used then fetch ll & rr.

- ``-baselines`` - lower and upper limits on baselines [ED].

- ``-ra`` - fetch only data with RA baselines.

- ``-user`` - user of RA DB.

- ``-password`` - password to user.

Notes:

- Data (``u``, ``v``, ``flux`` or ``flux_limit``, ``sigma_flux``) is saved to specified file.

- Optionally, one can fit fetched data using ``-mcmc`` or ``-leastsq`` parameters. But it is not tested yet:)

License
-------

Copyright 2014 Ilya Pashchenko.

``uvmod`` is free software made available under the MIT License. For details
see the LICENSE file.
