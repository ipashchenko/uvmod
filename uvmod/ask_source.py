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

# TODO: Plot RR&LL with different colors!

dtype_converter_dict = {'integer': 'int', 'smallint': 'int', 'character': '|S',
                        'character varying': '|S', 'real': '<f8',
                        'timestamp without time zone': np.object}


def dtype_converter(data_type, char_length):
    """
    Converts psycopg2 data types to python data types.
    :param data_type:
        Psycopg2 data type.
    :param char_length:
        If not ``None``, then shows char length.
    :return:
    """
    result = dtype_converter_dict[data_type]
    if char_length:
        result += str(char_length)

    return result


def get_source_array_from_dbtable(source, band, host='odin.asc.rssi.ru',
                                  port='5432', db='ra_results', user='guest',
                                  password='WyxRep0Dav',
                                  table_name='pima_observations'):
    """
    Function that returns numpy structured array from user-specified db table.
    :param host:
    :param port:
    :param db:
    :param user:
    :param password:
    :param table_name:
    :return:
    """
    connection = psycopg2.connect(host=host, port=port, dbname=db,
                                  password=password, user=user)
    cursor = connection.cursor()
    # First know column names
    cursor.execute("select column_name, data_type, character_maximum_length from\
                   information_schema.columns where table_schema = \'public\'\
                   and table_name=\'" + table_name + "\'")
    result = cursor.fetchall()
    dtype = list()
    #column_names, data_types, char_lengths = zip(*result):
    for column_name, data_type, char_length in result:
        dtype.append((column_name, dtype_converter(data_type, char_length)))

    # Convert to numpy data types

    # Now read the table and put to structured array
    cursor.execute('SELECT *\
                    FROM pima_observations WHERE source = %(source)s AND\
                    (polar=%(ll)s OR polar=%(rr)s) AND band = %(band)s', \
                   {'source': source, 'band': band, 'll': 'LL', 'rr': 'RR'})
    table_data = cursor.fetchall()
    struct_array = np.zeros(len(table_data), dtype=dtype)
    for i, (column_name, data_type, char_length,) in enumerate(result):
        struct_array[column_name] = zip(*table_data)[i]

    return struct_array


def s_thr_from_obs_row(row, raise_ra=True, n_q=0.637, dnu=16. * 10 ** 6, n=2):
    """
    Function that calculates sigma of detection from structured array row.
    :param row:
        Row of 2D structured array. Actually, an object with __getitem__ method
        and corresponding keys.
    :return:
        Sigma for detection using upper and lower bands.
    """
    rt1 = row['st1']
    rt2 = row['st2']
    polar = row['polar']
    band = row['band'].upper()
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
                                                                 row['solint']))
    except TypeError:
        return None

    return result


if __name__ == '__main__':

    parser = \
        argparse.ArgumentParser(description="Plot source data from RA-survey",
                                epilog="Help me to develop it here:"
                                       " https://github.com/ipashchenko/uvmod")
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

    struct_array = get_source_array_from_dbtable(source, band)
    # Put u,v from lambda to E.D
    struct_array['u'] = utils.uv_to_ed(struct_array['u'],
                                       lambda_cm=utils.band_cm_dict[band])
    struct_array['v'] = utils.uv_to_ed(struct_array['v'],
                                       lambda_cm=utils.band_cm_dict[band])
    detections = list()
    ulimits = list()
    # Find detections & upper limits
    for row in struct_array:
        sigma = s_thr_from_obs_row(row)
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

