#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import (print_function)
import os
import sys
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import ra_uvfit
import warnings
import argparse
import numpy as np
try:
    from pylab import (errorbar, plot, savefig)
    is_pylab = True
except ImportError:
    is_pylab = False
try:
    from scipy.stats import uniform
    is_scipy = True
except ImportError:
    warnings.warn('Install ``scipy`` python package to use least squares'
                  ' estimates.')
    is_scipy = False
# my own emcee:)
sys.path.append('/home/ilya/work/emcee')
try:
    import emcee
    is_emcee = True
except ImportError:
    warnings.warn('Install ``emcee`` python package to use MCMC.')
    is_emcee = False
try:
    import triangle
    is_triangle = True
except ImportError:
    warnings.warn('Install ``triangle.py`` python package to draw beautiful'
                  ' corner plots of posterior PDF.')
    is_triangle = False


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

    # TODO: Refactor to function func(fname, tuple_of_dim, optional_tuple)
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
    print ("x, y, sy, xl, yl, syl")
    print(x, y, sy, xl, yl, syl)

    model_1d = ra_uvfit.Model_1d
    # If we are told to use LS
    if args.use_leastsq:
        lsq = ra_uvfit.LS_estimates(x, y, model_1d, sy=sy)
        p, pcov = lsq.fit(args.p0)
        print(p, pcov)

        if args.savefig:
            errorbar(x, y, sy, fmt='.k')
            errorbar(xl, yl, syl, fmt='.r', lolims=True)
            model_plot = model_1d(np.arange(1000.) * xmax / 1000.)
            plot(np.arange(1000.) * xmax / 1000., model_plot(p))
            print ("Saving figure to " + args.savefig)
            savefig(args.savefig)

        if args.savefile:
            print ("Saving data to " + args.savefile)
            np.savetxt(args.savefile, p)
            f_handle = file(args.savefile, 'a')
            if not np.isnan(pcov):
                np.savetxt(f_handle, pcov)
            f_handle.close()

    # If not => use MCMC
    else:
        # TODO: select number of components according to input
        lnpr_list = list()
        for i, max_p in enumerate(args.max_p):
            lnpr_list.append((uniform.logpdf, [0, args.max_p[i]], dict(),))
        lnprs = tuple(lnpr_list)
        lnpr = ra_uvfit.LnPrior(lnprs)
        lnpost = ra_uvfit.LnPost(x, y, model_1d, sy=sy, x_limits=xl,
                                 y_limits=yl, sy_limits=syl, lnpr=lnpr)

        # Using affine-invariant MCMC
        nwalkers = 250
        ndim = len(lnprs)
        if not args.p0:
            p0 = np.random.uniform(low=0., high=1., size=(nwalkers, ndim))
        else:
            p0 = emcee.utils.sample_ball(args.p0, args.std0, size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
        pos, prob, state = sampler.run_mcmc(p0, 250)
        sampler.reset()
        sampler.run_mcmc(pos, 500)

        # TODO: print info
        # TODO: put this to method(sampler, ndim, perc=95)
        par_list = list()
        for i in range(ndim):
            sample_vec = sampler.flatchain[::10, i]
            p_hdi_min, p_hdi_max = ra_uvfit.hdi_of_mcmc(sample_vec)
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
            print ("Saving data to " + args.savefile)
            np.savetxt(args.savefile, np.asarray(par_list))