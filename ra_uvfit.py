#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
import numpy as np
import math
#try:
#    import emcee
#except ImportError:
#    raise ImportError('Install ``emcee`` python package to proceed')
try:
    from scipy.special import erf
    from scipy.optimize import leastsq
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
        Instance of ``LnPrior`` class. Actually any callable that returns ln of
        the prior pdf for parameter vector.

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
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        try:
            self.sy = np.asarray(sy)
        except ValueError:
            self.sy = sy
        self.x_limits = np.asarray(x_limits)
        self.y_limits = np.asarray(y_limits)
        try:
            self.sy_limits = np.asarray(sy_limits)
        except ValueError:
            self.sy_limits = sy_limits

        # Various assertions on data consistency
        assert(self.x.ndim in [1, 2])
        assert(len(self.x) == len(self.y))
        if self.sy is not None:
            assert(len(self.y) == len(self.sy))

        # Choose method for model calculation here
        if self.x.ndim == 1:
            self._model = Model_1d
        elif self.x.ndim == 2:
            self._model = Model_2d

        self._lnprob = list()
        self._lnprob.append(LnProbDetections(x, y, self._model, sy=sy))

        # If we have any data on upper limits
        if (self.x_limits is not None) or (self.y_limits is not None) or (self.sy_limits is not None):
            # Then we must have at least x & y for upper limits
            assert(self.x_limits is not None and self.y_limits is not None)
            # Assert equal dimensionality for x of detections and limits
            assert(self.x.ndim == self.x_limits.ndim)

            self._lnprob.append(LnProbUlimits(x_limits, y_limits, self._model,
                                              sy=sy_limits))

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
        raise NotImplementedError


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
        super(LnProbUlimits, self).__init__(x, y, model, sy=sy)

    def _lnprob1(self, p):
        return (np.log(0.5 * (1. + erf((self.y - self.model(p)) /
                (math.sqrt(2.) * self.sy))))).sum()

    def _lnprob2(self, p):
        return (np.log(0.5 * (1. + erf((self.y - self.model(p)) /
                (math.sqrt(2.) * p[-1]))))).sum()


class LnPrior(object):
    """
    Class that represents prior pdf for parameters.

    :param lnprs:
        Tuple of tuples (callable, args, kwargs,) where args & kwargs -
        additional arguments to callable. Each callable is called callable(p,
        *args, **kwargs).

        Example:
            ((scipy.stats.norm.logpdf, [mu, s], dict(),),
            (scipy.stats.beta.logpdf, [alpha, beta], dict(),),
            (scipy.stats.uniform.logpdf, [a, b - a], dict(),),
            (scipy.stats.lognorm.logpdf, [i don't know:)], dict,),)

        First tuple will result in calling: scipy.stats.norm.logpdf(x, mu, s)

    """
    def __init__(self, lnprs):
        self.lnprs = [_function_wrapper(func, args, kwargs) for func, args,
                      kwargs in lnprs]

    def __call__(self, p):
        """
        Returns ln of prior pdf for given parameter vector.
        """
        return sum([self.lnprs[i](p_) for i, p_ in enumerate(p)])


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("uvmod: Exception while calling your prior pdf:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


# TODO: implement sy being initial guess for uncertainty
class LS_estimates(object):
    """
    Class that implements estimates of parameters via least squares method.
    """
    def __init__(self, x, y, sy=None):
        # if sp is None:
        #     raise ImportError('scipy')
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

        p = leastsq(residuals, [0., 0.], args=(self.x, self.y, self.sy,))[0]
        sigma = math.sqrt(-1. / (2. * p[0]))
        amp = math.exp(p[1])

        return amp, sigma

    def fit_2d(self):
        """
        LS for 2D data.

        Fitting model log(y) = a * x ** 2 + b
        """
        raise NotImplementedError()


if __name__ == '__main__':

    # Generate 1d-data for given model: 2. * exp(-x ** 2. / (2. * 0.09))
    print("Generating 1-d data with amp=2, sigma=0.3")
    p = [2, 0.3]
    x = np.array([0., 0.1, 0.2, 0.4])
    model = Model_1d(x)
    y = model([2., 0.3]) + np.random.normal(0, 0.2, size=4)
    sy = np.random.normal(0, 0.2, size=4)
    #errorbar(x, y, sy, fmt='.k')
    xl = np.array([0.35, 0.45])
    yl = np.array([1., 0.5])
    syl = np.random.normal(0, 0.2, size=2)
    print(x)
    print(y)
    print(sy)

    # Testing ``LnLike``
    print("Testing ``LnLike``")
    lnlike = LnLike(x, y, sy=sy, x_limits=xl, y_limits=yl, sy_limits=syl)
    lnlike._lnprob[1].__call__(p)
    lnlike(p)

    # Testing ``LS_estimates``
    print("Testing ``LS_estimates``")
    lsq = LS_estimates(x, y, sy=sy)
    amp, sigma = lsq.fit_1d()
    print ("amp = " + str(amp), "sigma = " + str(sigma))



   # parser = argparse.ArgumentParser()

   # parser.add_argument('-a', action='store_true', default=False,
   # dest='use_archive_ftp', help='Use archive.asc.rssi.ru ftp-server for\
   # FITS-files')

   # parser.add_argument('-asc', action='store_const', dest='remote_dir',
   # const='/', help='Download asc-correlator FITS-files')

   # parser.add_argument('-difx', action='store_const', dest='remote_dir',
   # const='/quasars_difx/', help='Download difx-correlator FITS-files')

   # parser.add_argument('exp_name', type=str, help='Name of the experiment')
   # parser.add_argument('band', type=str, help='Frequency [c,k,l,p]')
   # parser.add_argument('refant', type=str, default='EFLSBERG', help='Ground antenna', nargs='?')

   # args = parser.parse_args()

   # if not args.remote_dir:
   #     sys.exit("Use -asc/-difx flags to select archive's fits-files")

   # # Sampling posterior density of parameters
   # lnpost = LnPost(x, y, sy=sy, x_limits=x_limits, y_limits=y_limits,
   #                 sy_limits=sy_limits, lnpr=lnpr, args=None,
   #                 kwargs=None)

   # # Using affine-invariant MCMC
   # nwalkers = 250
   # ndim = ndim
   # # Find p0 approx. using leastsq
   # p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))

   # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
   # pos, prob, state = sampler.run_mcmc(p0, 250)
   # sampler.reset()

   # sampler.run_mcmc(pos, 500)
