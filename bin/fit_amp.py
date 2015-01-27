#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import (print_function)
import os
import sys
import warnings
import argparse
import numpy as np
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import stats
from uvmod import models
from uvmod import plotting
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
        argparse.ArgumentParser(description='Fit simple models in uv-plane',
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
    parser.add_argument('-jitter', action='store_true', dest='jitter',
                        default=False, help='- model jitter?')
    parser.add_argument('-outliers', action='store_true', dest='outliers',
                        default=False, help='- model outliers?')
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
                                       ' or histograms (coming soon)')
    parser.add_argument('-savemodfig', action='store', nargs='?',
                        default=None, metavar='path to file', type=str,
                        help='- file to save plots of model vs data')
    parser.add_argument('-savefile', action='store', nargs='?', default=None,
                        metavar='path to file', type=str, help='- file to save'
                                                               ' parameters')

    args = parser.parse_args()

    if args.use_leastsq and (not args.p0):
        sys.exit("Use -p0 flag to specify the list of starting values for"
                 " minimization!")
    if args.use_leastsq and args.std0:
        print("Specified flag -std0 won't be used in routine!")
    # FIXME: In fact we can. Use MLE. Put -LnLike to minimization routine!
    if args.use_leastsq and args.jitter:
        print("Can't model jitter in LSQ!")
    # FIXME: In fact we can. Use MLE. Put -LnLike to minimization routine!
    if args.use_leastsq and args.outliers:
        print("Can't model outliers in LSQ!")
    if args.path_to_ulimits and args.outliers:
        sys.exit("Outliers with upper limits coming soon!")
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

    jitter_n = {True: 1, False: 0}
    outliers_n = {True: 3, False: 0}

    # Loading data
    # Pre-initialize in case of no uncertainties supplied
    sy, xl, yl, syl = [None] * 4
    # TODO: Refactor to function func(fname, tuple_of_dim, optional_tuple)
    if not args.use_2d:
        model = models.Model_1d
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
        # Pre-initialize in case of no uncertainties supplied. ``None`` values
        # will be used later in ploting functions
        xl1, xl2 = [None] * 2
        # Choose Model class to use
        print("We get " + str(len(args.p0)) + " parameters")
        n_gauss_pars = len(args.p0) - outliers_n[args.outliers] - jitter_n[args.jitter]
        print("Gauss function has number of parameters :")
        print(n_gauss_pars)
        if n_gauss_pars== 2:
            model = models.Model_2d_isotropic
            print("Will use isotropic gaussian model")
        elif n_gauss_pars == 4:
            model = models.Model_2d_anisotropic
            print("Will use anisotropic gaussian model")
        else:
            raise Exception("Only 2 or 4 parameters for 2D case!")

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

    # Print data
    if args.use_2d:
        print("=========== 2D ==============")
    else:
        print("=========== 1D ==============")
    print("Predictors of detections: ")
    print(x)
    print("Detection values: ")
    print(y)
    if sy is not None:
        print("Detection uncertainties: ")
        print(sy)
    if args.path_to_ulimits is not None:
        print("Predictors of upper limits: ")
        print(xl)
        print("Upper limits: ")
        print(yl)
        if syl is not None:
            print("Upper limits uncertainties: ")
            print(syl)
            print("============================")

    # Find max argument for plotting
    try:
        xmax = max((np.vstack((x, xl))).flatten())
    # If no data on limits
    except ValueError:
        xmax = max(x.flatten())

    # If we are told to use LSQ
    if args.use_leastsq:
        lsq = stats.LS_estimates(x, y, model, sy=sy)
        p, pcov = lsq.fit(args.p0)

        # Saving best fit params and covariance matrix
        if args.savefile:
            print ("Saving data to " + args.savefile)
            np.savetxt(args.savefile, p)
            f_handle = file(args.savefile, 'a')
            if not np.isnan(pcov).any():
                np.savetxt(f_handle, pcov)
            f_handle.close()

        # Plotting model vs. data
        if args.savemodfig:
            if args.use_2d:
                plotting.plot_all(p, x1=x1, x2=x2, y=y, sy=sy, ux1=xl1, ux2=xl2,
                                  uy=yl, outfile=args.savemodfig)
            else:
                errorbar(x, y, sy, fmt='.k')
                errorbar(xl, yl, syl, fmt='.r', lolims=True)
                model_plot = model(np.arange(1000.) * xmax / 1000.)
                plot(np.arange(1000.) * xmax / 1000., model_plot(p))
                print ("Saving figure to " + args.savemodfig)
                savefig(args.savemodfig)

    # If didn't told use LSQ => use MCMC to sample posterior
    else:
        lnpr_list = list()
        for i, max_p in enumerate(args.max_p):
            lnpr_list.append((uniform.logpdf, [0, args.max_p[i]], dict(),))
        lnprs = tuple(lnpr_list)
        lnpr = stats.LnPrior(lnprs)
        lnpost = stats.LnPost(x, y, model, sy=sy, x_limits=xl, y_limits=yl,
                              sy_limits=syl, lnpr=lnpr, jitter=args.jitter,
                              outliers=args.outliers)

        # Using affine-invariant MCMC
        nwalkers = 250
        ndim = len(lnprs)
        if not args.p0:
            p0 = np.random.uniform(low=0., high=1., size=(nwalkers, ndim))
        else:
            p0 = emcee.utils.sample_ball(args.p0, args.std0, size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
        print("Burning-in...")
        pos, prob, state = sampler.run_mcmc(p0, 150)
        sampler.reset()
        print("Sampling posterior...")
        sampler.run_mcmc(pos, 200)

        # Calculate mean and 95% HDI interval of parameter's posterior
        par_list = list()
        for i in range(ndim):
            sample_vec = sampler.flatchain[::10, i]
            p_hdi_min, p_hdi_max = stats.hdi_of_mcmc(sample_vec)
            p_mean = np.mean(sample_vec)
            par_list.append([p_hdi_min, p_mean, p_hdi_max])
            par_array = np.asarray(par_list)

        # Save mean and 95% HDI interval of parameter's posterior
        if args.savefile:
            print ("Saving data to " + args.savefile)
            np.savetxt(args.savefile, np.asarray(par_list))

        # Visualize with triangle_plot.py
        if args.savefig:
            # If ``triangle_plot.py`` is install use it for corner plot of
            # parameters posterior
            if is_triangle:
                figure = triangle.corner(sampler.flatchain[::10, :])
                print ("Saving figure to " + args.savefig)
                figure.savefig(args.savefig)
            # Plot histogram stuff wo triangle
            else:
                print("Can't plot posterior without triangle_plot.py!")

        # Plot model vs. data
        if args.savemodfig:
            # 2D case
            if args.use_2d:
                print(par_array[:n_gauss_pars], x1, x2, y, sy, xl1, xl2, yl)
                plotting.plot_all(par_array[:n_gauss_pars, 1], x1=x1, x2=x2, y=y,
                                  sy=sy, ux1=xl1, ux2=xl2, uy=yl,
                                  outfile=args.savemodfig)
            # 1D case
            else:
                errorbar(x, y, sy, fmt='.k')
                errorbar(xl, yl, syl, fmt='.r', lolims=True)
                model_plot = model(np.arange(1000.) * xmax / 1000.)
                plot(np.arange(1000.) * xmax / 1000.,
                     model_plot(par_array[:n_gauss_pars, 1]))
                print ("Saving figure to " + args.savemodfig)
                savefig(args.savemodfig)
