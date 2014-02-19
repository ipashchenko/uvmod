#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
import numpy as np
import math
try:
    import emcee
except ImportError:
    raise ImportError('Install ``emcee`` python package to proceed')
try:
    import scipy as sp
except ImportError:
    sp = None


class LnPost(object):
    """
    Class that represents posterior density.

    :param x:
        uv-radius (for 1D) or coordinates in uv-plane (for 2D models).

    :param y:
        Visibility amplitudes.

    :param sy (optional):
        Std of visibility amplitude measurements.

    :param x_limits (optional):
        uv-radius (for 1D) or coordinates in uv-plane (for 2D models)
        for upper limits.

    :param y_limits (optional):
        Visibility amplitudes for upper limits.

    :param sy_limits (optional):
        Std of upper limits visibility amplitude measurements.

    :param lnpr:
        Callable prior distribution.

    :param args (optional):
        Positional arguments for prior density callable.

    :param args (optional):
        Keyword arguments for prior density callable.
    """
    def __init__(self, x, y, sy=None, x_limits=None, y_limits=None,
                 sy_limits=None, lnpr=None, args=None, kwargs=None):

        if lnpr is None:
            raise Exception('Provide ``lnpr`` keyword argument')

        self._lnpr = lnpr
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        self.args = args
        self.kwargs = kwargs
        self._lnlike = LnLike(x, y, sy=sy, x_limits=x_limits, y_limits=y_limits,
                              sy_limits=sy_limits)

    def lnpr(self, p):
        return self._lnpr(p, *self.args, **self.kwargs)

    def lnlike(self, p):
        return self._lnlike.__call__(p)

    def __call__(self, p):
        return self.lnlike(p) + self.lnpr(p)


# Use ``Model`` class subclasses for strategy pattern? Or 2 methods of
# ``LnLike`` class? Better take models in separate class
# TODO: Add t-distribution noise model
# TODO: Account for outliers in data
class LnLike(object):
    """
    Class that represents Likelihood function.

    :param x:
        uv-radius (for 1D) or coordinates in uv-plane (for 2D models).

    :param y:
        Visibility amplitudes.

    :param sy (optional):
        Std of visibility amplitude measurements.

    :param x_limits (optional):
        uv-radius (for 1D) or coordinates in uv-plane (for 2D models)
        for upper limits.

    :param y_limits (optional):
        Visibility amplitudes for upper limits.

    :param sy_limits (optional):
        Std of upper limits visibility amplitude measurements.

    """
    def __init__(self, x, y, sy=None, x_limits=None, y_limits=None, sy_limits=None):
        # If x is 2D - (u,v) then use np.at_least_2d?
        self.x = np.array(x)
        self.y = np.array(y)
        try:
            self.sy = np.array(sy)
        except ValueError:
            self.sy = sy
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.sy_limits = sy_limits

        # Various assertions on data consistency
        assert(self.x.dim in [1, 2])
        assert(len(self.x) == len(self.y))
        if self.sy is not None:
            assert(len(self.y) == len(self.sy))

        # Choose method for model calculation here
        if self.x.dim == 1:
            self._model = Model_1d
        elif self.x.dim == 2:
            self._model = Model_2d

        self._lnprob = list()
        self._lnprob.append(LnProbDetections(x, y, self._model, sy=sy))

        # If we have any data on upper limits
        if (self.x_limits is not None) or (self.y_limits is not None) or (self.sy_limits is not None):
            # Then we must have at least x & y for upper limits
            assert(self.x_limits is not None and self.y_limits is not None)
            # Assert equal dimensionality for x of detections and limits
            assert(np.shape(self.x)[1] == np.shape(self.x_limits)[1])

            self._lnprob.append(LnProbUlimits(x_limits, y_limits, self._model,
                                              sy=sy))

# TODO: Should i try to use try/except?
    def __call__(self, p):
        return sum([lnprob(p) for lnprob in self._lnprob])


class Model(object):
    """
    Basic class that implements models.
    """
    def __init__(self, x):
        self.x = x

    def __call__(self, p):
        pass


class Model_1d(Model):
    def __call__(self, p):
        return p[0] * np.exp(-self.x ** 2 / (2. * p[1] ** 2))


class Model_2d(Model):
    def __call__(self, p):
        raise NotImplementedError('Coming soon!')


class LnProb(object):
    """
    Basic class that calculates the probability of parameters given
    data (i.e. likelihood).

    :param model:
        Instance of ``Model`` subclass.
    """
    def __init__(self, x, y, model, sy=None):
        self.x = x
        self.y = y
        self.sy = sy
        self.model = model(x)

        # TODO: Put logic to method?
        if self.sy is None:
            self.lnprob = self._lnprob2
        else:
            self.lnprob = self._lnprob1

    def __call__(self, p):
        return self.lnprob(p)

    def _lnprob1(self, p):
        """
        With estimated uncertainties.
        """
        raise NotImplementedError()

    def _lnprob2(self, p):
        """
        Without estimated uncertainties.
        """
        raise NotImplementedError()


class LnProbDetections(LnProb):
    """
    Likelihood for detections.
    """
    def __init__(self, x, y, model, sy=None):
        super(LnProbDetections, self).__init__(x, y, model, sy=sy)

    def _lnprob1(self, p):
        return (np.log(2. * math.pi * self.sy ** 2) -
               (self.y - self.model(p)) / (2. * self.sy ** 2)).sum()

    def _lnprob2(self, p):
        return (np.log(2. * math.pi * p[-1] ** 2) -
               (self.y - self.model(p)) / (2. * p[-1] ** 2)).sum()


class LnProbUlimits(LnProb):
    """
    Lnlikelihood for upper limits.
    """
    def __init__(self, x, y, model, sy=None):
        super(LnProbDetections, self).__init__(x, y, model, sy=sy)

    def _lnprob1(self, p):
        return (np.log(0.5 * (1. + sp.erf((self.y - self.model(p)) /
                (math.sqrt(2.) * self.sy))))).sum()

    def _lnprob2(self, p):
        return (np.log(0.5 * (1. + sp.erf((self.y - self.model(p)) /
                (math.sqrt(2.) * p[-1]))))).sum()


class LnPrior(object):
    """
    Class that represents prior pdf for parameters.
    """
     def __init__(self):
         pass


# TODO: implement sy being initial guess for uncertainty
class LS_estimates(object):
    """
    Class that implements estimates of parameters via least squares method.
    """
    def __init__(self, x, y, sy=None):
       if sp is None:
           raise ImportError('scipy')
       self.x = x
       self.y = y
       self.sy = sy

    def fit_1d(self):
       """
       LS for 1D data.

       Fitting model log(y) = a * x ** 2 + b
       """

       def residuals(p, x, y, sy):
           return (np.log(y) - p[0] * x ** 2. - p[1]) / (sy / y)

       result = sp.optimize.leastsq()

    def fit_2d(self):
        """
        LS for 2D data.

        Fitting model log(y) = a * x ** 2 + b
        """
        raise NotImplementedError()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', action='store_true', default=False,
    dest='use_archive_ftp', help='Use archive.asc.rssi.ru ftp-server for\
    FITS-files')

    parser.add_argument('-asc', action='store_const', dest='remote_dir',
    const='/', help='Download asc-correlator FITS-files')

    parser.add_argument('-difx', action='store_const', dest='remote_dir',
    const='/quasars_difx/', help='Download difx-correlator FITS-files')

    parser.add_argument('exp_name', type=str, help='Name of the experiment')
    parser.add_argument('band', type=str, help='Frequency [c,k,l,p]')
    parser.add_argument('refant', type=str, default='EFLSBERG', help='Ground antenna', nargs='?')

    args = parser.parse_args()

    if not args.remote_dir:
        sys.exit("Use -asc/-difx flags to select archive's fits-files")

    # Sampling posterior density of parameters
    lnpost = LnPost(x, y, sy=sy, x_limits=x_limits, y_limits=y_limits,
                    sy_limits=sy_limits, lnpr=lnpr, args=None,
                    kwargs=None)

    # Using affine-invariant MCMC
    nwalkers = 250
    ndim = ndim
    # Find p0 approx. using leastsq
    p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 250)
    sampler.reset()

    sampler.run_mcmc(pos, 500)



