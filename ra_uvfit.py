#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
import numpy as np
import math
try:
    from pylab import (errorbar, plot, savefig)
    is_pylab = True
except ImportError:
    is_pylab = False
# my own emcee:)
sys.path.append('/home/ilya/work/emcee')
try:
    import emcee
    is_emcee = True
except ImportError:
    raise ImportWarning('Install ``emcee`` python package to use MCMC.')
    is_emcee = False
try:
    import triangle
    is_triangle = True
except ImportError:
    raise ImportWarning('Install ``triangle`` python package to draw beautiful'
                        ' corner plots of posterior PDF.')
    is_triangle = False
try:
    from scipy.special import erf
    from scipy.optimize import leastsq
    from scipy.stats import uniform
    is_scipy = True
except ImportError:
    raise ImportWarning('Install ``scipy`` python package to use least squares'
                        ' estimates.')
    is_scipy = False


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
        if not is_scipy:
            raise ImportError("Install scipy to use " + str(self.__class__))

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


# TODO: add method ``fit`` that call 1d or 2d methods depending on ``x``
class LS_estimates(object):
    """
    Class that implements estimates of parameters via least squares method.

    The algorithm uses the Levenberg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm. It is
    almost scipy.optimize.curve_fit but with different arguments of function.

    :param x:
        Vector of explanatory variable.

    :param y:
        Vector of response variable.

    :param sy (optional):
        Vector of uncertainties. If not ``None`` then this vector will be used
        as relative weights in the least-squares problem. (default: ``None``)

    """

    def _weighted_residuals(cls, p, y, sy, model):
        return (y - model(p)) / sy

    def _residuals(cls, p, y, model):
        return (y - model(p))

    def __init__(self, x, y, sy=None):
        self.x = x
        self.y = y
        self.sy = sy

    def fit_1d(self, p0=None):
        """
        LS for 1D data.

        Fitting model log(y) = a * x ** 2 + b

        :param p0:
            The starting estimate for the minimization.

        :return:
            Optimized vector of parameters and it's covariance matrix.
        """

        if not is_scipy:
            raise ImportError("Install ``scipy`` to use ``fit_1d`` method of "
                              + str(self.__class__))
        if p0 is None:
            raise Exception("Define starting estimate for minimization!")

        model = Model_1d(self.x)

        if self.sy is None:
            func, args = self._residuals, (self.y, model,)
        else:
            func, args = self._weighted_residuals, (self.y, self.sy, model)

        fit = leastsq(func, p0, args=args, full_output=True)
        (p, pcov, infodict, errmsg, ier) = fit

        if ier not in [1, 2, 3, 4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        if (len(self.y) > len(p0)) and pcov is not None:
            # Residual variance
            s_sq = (func(p, *args) ** 2.).sum() / (len(self.y) - len(p0))
            pcov *= s_sq
        else:
            pcov = np.inf

        if p[1] < 0:
            p[1] *= -1.

        return p, pcov

    def fit_2d(self):
        """
        LS for 2D data.

        Fitting model log(y) = a * x ** 2 + b
        """
        raise NotImplementedError()


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    """
    Highest density interval of sample.
    """

    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    return hdi_min, hdi_max


if __name__ == '__main__':

    parser =\
        argparse.ArgumentParser(description="Fit simple models in uv-plane",
                                epilog="Help me to develop it here:"
                                       " https://github.com/ipashchenko/uvmod")

    parser.add_argument('-leastsq', action='store_true', dest='use_leastsq',
                        default=False,
                        help='- use scipy.optimize.leastsq for analysis of'
                             ' detections')
    parser.add_argument('-p0', action='store', dest='p0', nargs='+',
                        default=None, type=float, help='- starting estimates'
                                                       ' for the minimization'
                                                       ' or center of initial'
                                                       ' ball for MCMC')
    parser.add_argument('-std0', action='store', dest='std0', nargs='+',
                        default=None, type=float, help='- stds of initial ball'
                                                       ' for MCMC')
    parser.add_argument('-2d', action='store_true', dest='use_2d',
                        default=False, help='- use 2D-fitting?')
    parser.add_argument('path_to_detections', type=str, metavar='detections',
                        help='- path to file with detections data')
    parser.add_argument('path_to_ulimits', nargs='?', metavar='upper limits',
                        default=None, type=str, help=' - path to file with'
                                                     ' upper limits data')
    parser.add_argument('-max_p', action='store', nargs='+', default=None,
                        type=float, help='- maximum values of uniform prior'
                                         ' distribution for parameters')
    parser.add_argument('-savefig', action='store', nargs='?',
                        default=None, metavar='path to file',
                        type=str, help='- file to save plots of posterior'
                                       ' PDF (if ``triangle.py`` is installed)'
                                       ' or histograms (coming soon). If'
                                       ' -leastsq flag is set then plot data'
                                       ' and best model')
    parser.add_argument('-savefile', action='store', nargs='?', default=None,
                        metavar='path to file', type=str, help='- file to save'
                                                               ' results')

    args = parser.parse_args()

    if args.use_2d:
        raise NotImplementedError("Coming soon!")

    if args.use_leastsq and (not args.p0):
        sys.exit("Use -p0 flag to specify the list of starting values for"
                 " minimization!")

    if args.use_leastsq and args.std0:
        print("Specified flag -std0 won't be used in routine!")

    if (not args.use_leastsq) and (not args.max_p):
        sys.exit("Use -max_p flag to specify the list of maximum values of"
                 " parameters in uniform prior distributions")

    if (not args.use_leastsq) and args.p0 and (not args.std0):
        sys.exit("Use -std0 flag to specify value of std for initial parametes"
                 " ball")

    if (not args.use_leastsq) and args.std0 and (not args.p0):
        sys.exit("Use -p0 flag to specify the center of ball for initial"
                 " parameters values")

    if (not args.use_leastsq) and (not args.p0) and (not args.std0):
        print("Use -p0 flag to specify the center of ball for initial"
              " parameters values!")
        print("Use -std0 flag to specify value of std for that ball!")

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

    xmax = max(np.hstack((x, xl)))
    print (x, y, sy, xl, yl, syl)

    # If we are told to use LS
    if args.use_leastsq:
        lsq = LS_estimates(x, y, sy=sy)
        p, pcov = lsq.fit_1d(args.p0)

        if args.savefig:
            errorbar(x, y, sy, fmt='.k')
            errorbar(xl, yl, syl, fmt='.r', lolims=True)
            model_plot = Model_1d(np.arange(1000.) * xmax / 1000.)
            plot(np.arange(1000.) * xmax / 1000., model_plot(p))
            savefig(args.savefig)

        if args.savefile:
            np.savetxt(args.savefile, p)
            f_handle = file(args.savefile, 'a')
            np.savetxt(f_handle, pcov)

    # If not => use MCMC
    else:
        lnprs = ((uniform.logpdf, [0, args.max_p[0]], dict(),),
                 (uniform.logpdf, [0, args.max_p[1]], dict(),),)
        lnpr = LnPrior(lnprs)
        lnpost = LnPost(x, y, sy=sy, x_limits=xl, y_limits=yl, sy_limits=syl,
                        lnpr=lnpr)

        # Using affine-invariant MCMC
        nwalkers = 250
        ndim = 2
        if not args.p0:
            p0 = np.random.uniform(low=0., high=1., size=(nwalkers, ndim))
        else:
            p0 = emcee.utils.sample_ball(args.p0, args.std0, size=nwalkers)
        print(p0, np.shape(p0))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
        pos, prob, state = sampler.run_mcmc(p0, 250)
        sampler.reset()
        sampler.run_mcmc(pos, 500)

        # TODO: print info
        # TODO: put this to method(sampler, ndim, perc=95)
        par_list = list()
        for i in range(ndim):
            sample_vec = sampler.flatchain[::10, i]
            p_hdi_min, p_hdi_max = hdi_of_mcmc(sample_vec)
            p_mean = np.mean(sample_vec)
            par_list.append([p_hdi_min, p_mean, p_hdi_max])

        # Visualize with triangle.py
        if args.savefig:
            # If ``triangle.py`` is install use it for corner plot of posterior
            # PDF
            if is_triangle:
                figure = triangle.corner(sampler.flatchain[::10, :])
                print ("Saving figure to " + args.savefig)
                figure.savefig(args.savefig)
            # Plot histogram stuff wo triangle
            else:
                raise NotImplementedError("Coming soon!")

        if args.savefile:
            np.savetxt(args.savefile, np.asarray(par_list))
