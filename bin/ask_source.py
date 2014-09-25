import os
import sys
import math
import argparse
import warnings
import numpy as np
from scipy.stats import uniform
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import utils, models, stats, plotting
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


# TODO: Plot RA baselines with different colors!
if __name__ == '__main__':

    parser = \
        argparse.ArgumentParser(description="Plot source data from RA-survey",
                                epilog="Help me to develop it here:"
                                       " https://github.com/ipashchenko/uvmod")
    parser.add_argument('-pols', action='store', nargs='?', default=None,
                        metavar='pols', type=str, help='- None, rr, ll, rl,'
                                                       'lr')
    parser.add_argument('-ra', action='store_true', dest='ra_only',
                        default=False, help='Plot only RA baselines?')
    parser.add_argument('-savefig', action='store_true', dest='plot_to_file',
                        default=False, help='plot results?')
    parser.add_argument('-savefile', action='store_true', dest='save_to_file',
                        default=False, help='save results?')
    parser.add_argument('-source', action='store', nargs='?',
                        default=None, metavar='source name',
                        type=str, help='- source name to query [B1950]')
    parser.add_argument('-band', action='store', nargs='?',
                        default=None, metavar='band',
                        type=str, help='- k, c, l or p')
    parser.add_argument('-leastsq', action='store_true', dest='use_leastsq',
                        default=False,
                        help='- use scipy.optimize.leastsq for analysis of'
                             ' detections')
    parser.add_argument('-mcmc', action='store_true', dest='use_mcmc',
                        default=False,
                        help='- use MCMC for analysis of detections')
    parser.add_argument('-p0', action='store', dest='p0', nargs='+',
                        default=None, type=float, help='- starting estimates'
                                                       ' for the minimization'
                                                       ' or center of initial'
                                                       ' ball for MCMC')
    parser.add_argument('-std0', action='store', dest='std0', nargs='+',
                        default=None, type=float, help='- stds of initial ball'
                                                       ' for MCMC')
    parser.add_argument('-max_p', action='store', nargs='+', default=None,
                        type=float, help='- maximum values of uniform prior'
                                         ' distribution for parameters')
    parser.add_argument('-baselines', action='store', dest='baselines',
                        nargs='+', default=None, type=float,
                        help='- lower and upper range of baselines [ED]')
    parser.add_argument('-plot_model', action='store_true', dest='plot_model',
                        default=False, help='plot fitted model?')
    parser.add_argument('-user', action='store', nargs='?',
                        default=None, metavar='user',
                        type=str, help='- user in odin DB')
    parser.add_argument('-password', action='store', nargs='?',
                        default=None, metavar='password',
                        type=str, help='- password for user to odin DB')

    args = parser.parse_args()

    # Check DB connection parameters
    if not args.user:
        raise Exception("Use ``-user`` flag to supply a user in odin DB!")
    if not args.password:
        raise Exception("Use ``-password`` flag to supply a password to odin"
                        "DB!")
    # Check source and band
    if not args.source:
        raise Exception("Use ``-source`` flag to choose source!")
    if not args.band:
        raise Exception("Use ``-band`` flag to choose band!")

    # Check fitting parameters
    if args.use_leastsq and not args.p0:
        sys.exit("Use -p0 flag to specify the list of starting values for"
                 " minimization!")
    if args.use_leastsq and args.std0:
        print("Specified flag -std0 won't be used in routine!")
    if args.use_mcmc and not args.p0:
        sys.exit("Use -p0 flag to specify the center of ball for initial"
                 " parameters values")
    if args.use_mcmc and not args.max_p:
        sys.exit("Use -max_p flag to specify the list of maximum values of"
                 " parameters in uniform prior distributions")
    if args.use_mcmc and not args.std0:
        sys.exit("Use -std0 flag to specify value of std for initial parametes"
                 " ball")

    host = 'odin.asc.rssi.ru'
    port = '5432'
    db = 'ra_results'
    user = args.user
    password = args.password
    table = 'pima_observations'
    # Default argument for plotting
    savefig = None
    source = args.source
    band = args.band

    struct_array = utils.get_source_array_from_dbtable(source, band, user=user,
                                                       password=password)

    if args.ra_only:
        struct_array = struct_array[np.where(np.logical_or(struct_array['st1']
                                                           == 'RADIO-AS',
                                                           struct_array['st2']
                                                           == 'RADIO-AS'))]


    if args.pols is not None:
        # Choosing polarization
        struct_array = struct_array[np.where(struct_array['polar']==
                                             args.pols.upper())]
    else:
        struct_array = struct_array[np.where(np.logical_or(struct_array['polar']
                                                           == 'LL',
                                                           struct_array['polar']
                                                           == 'RR'))]
    # Find detections & upper limits

    detections = list()
    ulimits = list()

    for row in struct_array:
        sigma = utils.s_thr_from_obs_row(row)
        if not sigma:
            continue
        if row['status'] == 'y':
            detections.append([row['u'], row['v'], row['snr'] * sigma, sigma])
        else:
            ulimits.append([row['u'], row['v'], 3. * sigma, sigma])

    detections = np.atleast_2d(detections)
    ulimits = np.atleast_2d(ulimits)

    # If have baseline ranges then use only data within
    if args.baselines:
        low, high = args.baselines
        print low, high
        print utils.band_cm_dict[band]
        low = utils.ed_to_uv(low, lambda_cm=utils.band_cm_dict[band])
        high = utils.ed_to_uv(high, lambda_cm=utils.band_cm_dict[band])
        print low, high
    else:
        low = -np.inf
        high = +np.inf


    print "Got ", detections.size, " detections!"
    try:
        xx = detections[:, :2]
        detections = detections[np.logical_and(np.sqrt(xx[:, 0] ** 2. +
                                                       xx[:, 1] ** 2.) > low,
                                               np.sqrt(xx[:, 0] ** 2. +
                                                       xx[:, 1] ** 2.) < high)]
        xx = detections[:, :2]
        y = detections[:, 2]
        sy = detections[:, 3]
        # convert to ED for plotting
        x1, x2 = utils.uv_to_ed(detections[:, :2],
                                lambda_cm=utils.band_cm_dict[band]).T
    # If no detections (detections.size = 0)
    except IndexError:
        # We need them as ``None`` to put all data in one plotting function.
        x1, x2, y, sy = [None] * 4

    print "Got ", ulimits.size, "upper limits!"
    try:
        uxx = ulimits[:, :2]
        ulimits = ulimits[np.logical_and(np.sqrt(uxx[:, 0] ** 2. +
                                                 uxx[:, 1] ** 2.) > low,
                                         np.sqrt(uxx[:, 0] ** 2. +
                                                 uxx[:, 1] ** 2.) < high)]
        uxx = ulimits[:, :2]
        uy = ulimits[:, 2]
        usy = ulimits[:, 3]
        # convert to ED for plotting
        ux1, ux2 = utils.uv_to_ed(ulimits[:, :2],
                                  lambda_cm=utils.band_cm_dict[band]).T
    # If no upper limits (ulimits.size = 0)
    except IndexError:
        # We need them as ``None`` to put all data in one plotting function.
        ux1, ux2, uy, usy = [None] * 4

    if args.save_to_file:
        np.savetxt(source + '_' + band + '_detections.txt', detections)
        np.savetxt(source + '_' + band + '_ulimits.txt', ulimits)
    if args.plot_to_file:
        savefig = source + '_' + band
    plotting.scatter_3d_errorbars(x1=x1, x2=x2, y=y, sy=sy, ux1=ux1, ux2=ux2,
                                  uy=uy, savefig=savefig)

    # Now fitting detections and upper limits
    # If we are told to use LSQ
    if args.use_leastsq:
        if not detections:
            raise Exception("Need detections for LSQ!")
        p0 = args.p0
        if len(p0) == 2:
            print "Fitting isotropic 2d-gaussian model to data!"
            model_to_use = models.Model_2d_isotropic
        elif len(p0) == 4:
            print "Fitting anisotropic 2d-gaussian model to data!"
            model_to_use = models.Model_2d_anisotropic
        else:
            raise Exception("Work only with single 2d gaussian model (2 or 4"
                            "pars)!")
        # Put mas in args.p0 to uv for major axis.
        p0[1] = 1./ (2. * math.pi * p0[1] * utils.mas_to_rad)
        print "xx : ", xx
        print "y : ", y
        print "p0 after conv to uv spatial frequencies : ", p0
        lsq = stats.LS_estimates(xx, y, model_to_use, sy=sy)
        p, pcov = lsq.fit(p0)
        print p, pcov
        # Put from uv to mas for major axis.
        p_mas = p.copy()
        p_mas[1] = 1. / (2. * math.pi * p[1] * utils.mas_to_rad)
        try:
            pcov = np.atleast_2d(pcov)
        except:
            print "Sucked up with covariance estimation!"
        pcov[1, :] = 1. / (2. * math.pi * pcov[1, :] * utils.mas_to_rad)
        pcov[:, 1] = 1. / (2. * math.pi * pcov[:, 1] * utils.mas_to_rad)

        print "LSQ gave p: ", p_mas
        print "LSQ gave pcov: ", pcov

        if args.plot_model:
            p[1] = utils.uv_to_ed(p[1], lambda_cm=utils.band_cm_dict[band])
            print "Plotting model with parameters : ", p
            plotting.plot_all(p, x1, x2, y, sy, n=100)