
import os
import sys
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import utils
import math
import psycopg2
import numpy as np
import argparse
from plotting import scatter_3d_errorbars


def s_thr_from_obs_row(row, raise_ra=True, n_q=0.637, dnu=16. * 10 ** 6, n=2):
    """
    Function that calculates sigma of detection from structured array row.
    :param row:
        Tuple of one row from query.
    :return:
        Sigma for detection using upper and lower bands.
    """
    rt1 = row[0]
    rt2 = row[1]
    polar = row[2]
    band = row[3].upper()
    try:
        SEFD_rt1 = utils.SEFD_dict[rt1][band.upper()][polar[0]]
    except KeyError:
        #raise Exception("There's no entry for " + rt1 + " in SEFD dictionary!")
        return None
    except TypeError:
        raise Exception("There's no SEFD data for " + rt1 + " !")
    try:
        SEFD_rt2 = utils.SEFD_dict[rt2][band.upper()][polar[1]]
    except KeyError:
        #raise Exception("There's no entry for " + rt2 + " in SEFD dictionary!")
        return None
    except TypeError:
        raise Exception("There's no SEFD data for " + rt2 + " !")

    try:
        result = (1. / n_q) * math.sqrt((SEFD_rt1 * SEFD_rt2) / (n * dnu *
                                                                 row[4]))
    except TypeError:
        return None

    return result

if __name__ == '__main__':

    parser = \
        argparse.ArgumentParser(description="Plot source data from RA-survey",
                                epilog="Help me to develop it here:"
                                       " https://github.com/ipashchenko/uvmod")

    parser.add_argument('-all', action='store_true', dest='plot_all_data',
                        default=False,
                        help='- plot data from all scans')
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

    connection = psycopg2.connect(database=db, user=user, password=password,
                                  host=host, port=port)
    cursor = connection.cursor()
    cursor.execute('SELECT st1, st2, polar, band, solint, snr, u, v, status\
                    FROM pima_observations WHERE source = %(source)s AND\
                    (polar=%(ll)s OR polar=%(rr)s) AND band = %(band)s',\
                    {'source': source, 'band': band, 'll': 'LL', 'rr': 'RR'})
    rows = cursor.fetchall()
    print len(rows), rows

    detections = list()
    ulimits = list()
    # Find detections & upper limits
    for row in rows:
        sigma = s_thr_from_obs_row(row)
        if not sigma:
            print sigma
            print "No SEFD data for ", row
            continue
        if row[8] == 'y':
            detections.append([row[6], row[7], row[5] * sigma, sigma])
        else:
            ulimits.append([row[6], row[7], 3. * sigma])

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
            x1 = detections_[:, 0]
            x2 = detections_[:, 1]
            y = detections_[:, 2]
            sy = detections_[:, 3]
    else:
        x1, x2, y, sy = [None] * 4
    if ulimits:
        try:
            ux1, ux2, uy = zip(ulimits)
        except ValueError:
            ulimits_ = np.atleast_2d(ulimits)
            ux1 = ulimits_[:, 0]
            ux2 = ulimits_[:, 1]
            uy = ulimits_[:, 2]
    else:
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

