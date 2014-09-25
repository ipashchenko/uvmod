import os
import sys
import math
import argparse
import numpy as np
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import utils, models, stats
from plotting import scatter_3d_errorbars, plot_all


# TODO: Plot RA baselines with different colors!
# TODO: Use only range of baselines for fitting! Otherwise it is useless...
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
    parser.add_argument('-p0', action='store', dest='p0', nargs='+',
                        default=None, type=float, help='- starting estimates'
                                                       ' for the minimization'
                                                       ' or center of initial'
                                                       ' ball for MCMC')
    parser.add_argument('-plot_model', action='store_true', dest='plot_model',
                        default=False, help='plot fitted model?')

    args = parser.parse_args()

    if not args.source:
        raise Exception("Use ``-source`` flag to choose source!")
    if not args.band:
        raise Exception("Use ``-band`` flag to choose band!")

    host='odin.asc.rssi.ru'
    port='5432'
    db='ra_results'
    user='guest'
    password='WyxRep0Dav'
    table='pima_observations'
    savefig = None
    source = args.source
    band = args.band

    struct_array = utils.get_source_array_from_dbtable(source, band)
    print struct_array
    # Put u,v from lambda to E.D
    #struct_array['u'] = utils.uv_to_ed(struct_array['u'],
    #                                   lambda_cm=utils.band_cm_dict[band])
    #struct_array['v'] = utils.uv_to_ed(struct_array['v'],
    #                                   lambda_cm=utils.band_cm_dict[band])

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
    detections = list()
    ulimits = list()

    # Find detections & upper limits
    for row in struct_array:
        sigma = utils.s_thr_from_obs_row(row)
        if not sigma:
            print sigma
            print "No SEFD data for ", row
            continue
        if row['status'] == 'y':
            detections.append([row['u'], row['v'], row['snr'] * sigma, sigma])
        else:
            ulimits.append([row['u'], row['v'], 3. * sigma])

    print "Detections: "
    print detections
    print "===================================="
    print "Upper limits: "
    print ulimits

    if detections:
        try:
            x1, x2, y, sy = zip(detections)
        except ValueError:
            detections_ = np.atleast_2d(detections)
            x1 = utils.uv_to_ed(detections_[:, 0],
                                lambda_cm=utils.band_cm_dict[band])
            x2 = utils.uv_to_ed(detections_[:, 1],
                                lambda_cm=utils.band_cm_dict[band])
            # Keep in uv not ED!
            xx = detections_[:, :2]
            y = detections_[:, 2]
            sy = detections_[:, 3]
    else:
        # We need them as ``None`` to put all data in one plotting function.
        x1, x2, y, sy = [None] * 4
    if ulimits:
        try:
            ux1, ux2, uy = zip(ulimits)
        except ValueError:
            ulimits_ = np.atleast_2d(ulimits)
            ux1 = utils.uv_to_ed(ulimits_[:, 0],
                                 lambda_cm=utils.band_cm_dict[band])
            ux2 = utils.uv_to_ed(ulimits_[:, 1],
                                 lambda_cm=utils.band_cm_dict[band])
            # Keep in uv not ED!
            uxx = ulimits_[:, :2]
            uy = ulimits_[:, 2]
    else:
        # We need them as ``None`` to put all data in one plotting function.
        ux1, ux2, uy = [None] * 3

    if args.save_to_file:
        if detections:
            detections = np.atleast_2d(detections)
            np.savetxt(source + '_' + band + '_detections.txt', detections)
        if ulimits:
            ulimits = np.atleast_2d(ulimits)
            np.savetxt(source + '_' + band + '_ulimits.txt', ulimits)
    if args.plot_to_file:
        savefig = source + '_' + band
    scatter_3d_errorbars(x1=x1, x2=x2, y=y, sy=sy, ux1=ux1, ux2=ux2, uy=uy,
                         savefig=savefig)

    # Now fitting detections and upper limits
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
            plot_all(p, x1, x2, y, sy, n=100)