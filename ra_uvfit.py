#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
import numpy as np
import math
from pylab import (errorbar, plot)
sys.path.append('/home/ilya/work/emcee')
try:
    import emcee
except ImportError:
    raise ImportError('Install ``emcee`` python package to proceed')
try:
    import triangle
except ImportError:
    raise ImportError('Install ``triangle`` python package to draw beautiful'
                      ' plots')
try:
    from scipy.special import erf
    from scipy.optimize import leastsq
    from scipy.stats import uniform
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
    def __init__(self, x, y, sy=None, x_limits=None, y_limits=None,
                 sy_limits=None):
        # If x is 2D - (u,v) then use np.at_least_2d?
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if sy is not None:
            self.sy = np.asarray(sy)
        else:
            self.sy = None
        if x_limits is not None:
            self.x_limits = np.asarray(x_limits)
            self.y_limits = np.asarray(y_limits)
        else:
            self.x_limits = x_limits
            self.y_limits = y_limits
        if sy_limits is not None:
            self.sy_limits = np.asarray(sy_limits)
        else:
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
        if (self.x_limits is not None) or (self.y_limits is not None) or\
                (self.sy_limits is not None):
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
        return (-0.5 * np.log(2. * math.pi * self.sy ** 2) -
               (self.y - self.model(p)) ** 2. / (2. * self.sy ** 2)).sum()

    def _lnprob2(self, p):
        return (-0.5 * np.log(2. * math.pi * p[-1] ** 2) -
               (self.y - self.model(p)) ** 2. / (2. * p[-1] ** 2)).sum()


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

    def fit_1d(self, p0=None):
        """
        LS for 1D data.

        Fitting model log(y) = a * x ** 2 + b

        :param p0 (optional):

            The starting estimate for the minimization. If ``None`` is given
            then use [0., 0.]. (default: ``None``)
        """

        if p0 is None:
            p0 = [0., 0.]

        def residuals(p, x, y, sy):
            return (np.log(y) - p[0] * x ** 2. - p[1]) / (sy / y)

        fit = leastsq(residuals, p0, args=(self.x, self.y, self.sy,),
                    full_output=True)
        (p, pcov, infodict, errmsg, ier) = fit

        if ier not in [1, 2, 3, 4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        if (len(y) > len(p0)) and pcov is not None:
            # Residual variance
            s_sq = (residuals(p, x, y, sy)**2).sum()/(len(y)-len(p0))
            pcov *= s_sq
        else:
            pcov = np.inf

        print(p, pcov)
        sigma = math.sqrt(-1. / (2. * p[0]))
        amp = math.exp(p[1])

        std_amp = amp * math.sqrt(pcov[1, 1])
        std_sigma = math.sqrt(-2. * p[0] * pcov[0, 0])

        return amp, sigma, std_amp, std_sigma

    def fit_2d(self):
        """
        LS for 2D data.

        Fitting model log(y) = a * x ** 2 + b
        """
        raise NotImplementedError()


if __name__ == '__main__':

    # # importing
    # from ra_uvfit import Model_1d, LnLike, LS_estimates, LnPrior, LnPost
    # from scipy.stats import uniform
    # import triangle

    # # Generate 1d-data for given model: 2. * exp(-x ** 2. / (2. * 0.09))
    # print("Generating 1-d data with amp=2, sigma=0.3")
    # p = [2, 0.3]
    # x = np.array([0., 0.1, 0.2, 0.4, 0.6])
    # model = Model_1d(x)
    # y = model([2., 0.3]) + np.random.normal(0, 0.1, size=5)
    # sy = np.random.normal(0.15, 0.025, size=5)
    # errorbar(x, y, sy, fmt='.k')
    # xl = np.array([0.5, 0.7])
    # yl = np.array([0.6, 0.2])
    # syl = np.random.normal(0.1, 0.03, size=2)
    # errorbar(xl, yl, syl, fmt='.r', lolims=True)
    # model_plot = Model_1d(np.arange(750) / 1000.)
    # plot(np.arange(750) / 1000., model_plot(p))
    # print(x)
    # print(y)
    # print(sy)

    # # Testing ``LnLike``
    # print("Testing ``LnLike``")
    # lnlike = LnLike(x, y, sy=sy, x_limits=xl, y_limits=yl, sy_limits=syl)
    # lnlike._lnprob[0].__call__(p)
    # lnlike._lnprob[1].__call__(p)
    # lnlike(p)

    # # Testing ``LS_estimates``
    # print("Testing ``LS_estimates``")
    # lsq = LS_estimates(x, y, sy=sy)
    # amp, sigma = lsq.fit_1d()
    # print ("amp = " + str(amp), "sigma = " + str(sigma))

    # # Testing ``LnPost``
    # lnprs = ((uniform.logpdf, [0, 10], dict(),),
    #          (uniform.logpdf, [0, 2], dict(),),)
    # lnpr = LnPrior(lnprs)
    # lnpost = LnPost(x, y, sy=sy, x_limits=xl, y_limits=yl, sy_limits=syl,
    #                 lnpr=lnpr)
    # assert(lnpost._lnpr(p) == lnpr(p))
    # assert(lnpost._lnlike(p) == lnlike(p))

    # # Using affine-invariant MCMC
    # nwalkers = 250
    # ndim = 2
    # p0 = np.random.uniform(low=0., high=1., size=(nwalkers, ndim))
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    # pos, prob, state = sampler.run_mcmc(p0, 250)
    # sampler.reset()
    # sampler.run_mcmc(pos, 500)
    # # Visualize with triangle.py
    # triangle.corner(sampler.flatchain[::10, :])

    parser =\
        argparse.ArgumentParser(description="Fit simple models in uv-plane",
                                epilog="Help me to develop it here:"
                                       " https://github.com/ipashchenko/uvmod")

    parser.add_argument('-leastsq', action='store_true', dest='use_leastsq',
                        default=False,
                        help='use scipy.optimize.leastsq for analysis of'
                             ' detections.')
    parser.add_argument('-2d', action='store_true', dest='use_2d',
                        default=False, help='Use 2D-fitting?')
    parser.add_argument('path_to_detections', type=str, metavar='detections',
                        help='path to file with detections data.')
    parser.add_argument('path_to_ulimits', nargs='?', metavar='upper limits',
                        default=None, type=str, help='path to file with upper'
                                                     ' limits data.')
    parser.add_argument('-max_amp', action='store', nargs='?', default=None,
                        type=float, help='maximum amplitude for uniform prior'
                                         ' distribution. If not given => use 10'
                                         ' x max(data)')
    parser.add_argument('-max_std', action='store', nargs='?', default=None,
                        type=float, help='maximum uncertainty for uniform prior'
                                         ' distribution. If not given => use'
                                         ' std(data)')
    parser.add_argument('-savefig', action='store', nargs='?',
                        default='uvmod_figure.png', metavar='path to file',
                        type=str, help='file to save corner plot of posterior'
                                       ' PDF. If not given =>'
                                       ' "uvmod_figure.py".')

    args = parser.parse_args()

    # if not args.remote_dir:
    #     sys.exit("Use -asc/-difx flags to select archive's fits-files")

    print(parser.parse_args())

    # TODO: refactor to function func(fname, tuple_of_dim, optional_tuple)
    # Pre-initialize in case of no uncertainties supplied
    xl, yl, sy, syl = [None] * 4
    if not args.use_2d:
        try:
            x, y, sy = np.loadtxt(args.path_to_detections, unpack=True)
        except ValueError:
            x, y = np.loadtxt(args.path_to_detections, unpack=True)
        if args.path_to_ulimits:
            try:
                xl, yl, syl = np.loadtxt(args.path_to_ulimits, unpack=True)
            except ValueError:
                xl, yl = np.loadtxt(args.path_to_ulimits, unpack=True)
    else:
        try:
            x1, x2, y, sy = np.loadtxt(args.path_to_detections, unpack=True)
        except ValueError:
            x1, x2, y = np.loadtxt(args.path_to_detections, unpack=True)
        x = np.column_stack((x1, x2,))
        if args.path_to_ulimits:
            try:
                xl1, xl2, yl, syl = np.loadtxt(args.path_to_ulimits,
                                               unpack=True)
            except ValueError:
                xl1, xl2, yl = np.loadtxt(args.path_to_ulimits, unpack=True)
            xl = np.column_stack((xl1, xl2,))

    print (x, y, sy, xl, yl, syl)

    # If no ranges for uniform priors are given => calculate them
    max_amp = args.max_amp or 10. * np.max(y)
    max_std = args.max_std or np.std(y)


    lnprs = ((uniform.logpdf, [0, max_amp], dict(),),
             (uniform.logpdf, [0, max_std], dict(),),)
    lnpr = LnPrior(lnprs)
    lnpost = LnPost(x, y, sy=sy, x_limits=xl, y_limits=yl, sy_limits=syl,
                    lnpr=lnpr)

    # Using affine-invariant MCMC
    nwalkers = 250
    ndim = 2
    p0 = np.random.uniform(low=0., high=1., size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 250)
    sampler.reset()
    sampler.run_mcmc(pos, 500)
    # Visualize with triangle.py
    figure = triangle.corner(sampler.flatchain[::10, :])
    print ("Saving figure to " + args.savefig)
    figure.savefig(args.savefig)







