import math
import psycopg2
import numpy as np


mas_to_rad = 4.8481368 * 1E-09
n_q = 0.637
vfloat = np.vectorize(float)
band_cm_dict = {'c': 6., 'l': 18., 'p': 94., 'k': 1.35 }
SEFD_dict = {'RADIO-AS': {'K': {'L': 46700., 'R': 36800},
                          'C': {'L': 11600., 'R': None},
                          'L': {'L': 2760., 'R': 2930.}},
             'GBT-VLBA': {'K': {'L': 23., 'R': 23.},
                          'C': {'L': 8., 'R': 8.},
                          'L': {'L': 10., 'R': 10.}},
             'EFLSBERG': {'C': {'L': 20., 'R': 20.},
                          'L': {'L': 19., 'R': 19.}},
             'YEBES40M': {'C': {'L': 160., 'R': 160.},
                          'L': {'L': None, 'R': None}},
             'ZELENCHK': {'C': {'L': 400., 'R': 400.},
                          'L': {'L': 300., 'R': 300.}},
             'EVPTRIYA': {'C': {'L': 44., 'R': 44.},
                          'L': {'L': 44., 'R': 44.}},
             'SVETLOE': {'C': {'L': 250., 'R': 250.},
                         'L': {'L': 360., 'R': 360.}},
             'BADARY': {'C': {'L': 200., 'R': 200.},
                        'L': {'L': 330., 'R': 330.}},
             'TORUN': {'C': {'L': 220., 'R': 220.},
                       'L': {'L': 300., 'R': 300.}},
             'ARECIBO': {'C': {'L': 5., 'R': 5.},
                         'L': {'L': 3., 'R': 3.}},
             'WSTRB-07': {'C': {'L': 120., 'R': 120.},
                          'L': {'L': 40., 'R': 40.}},
             'VLA-N8': {'C': {'L': None, 'R': None},
                        'L': {'L': None, 'R': None}},
             # Default values for KL
             'KALYAZIN': {'C': {'L': 150., 'R': 150.},
                          'L': {'L': 140., 'R': 140.}},
             'MEDICINA': {'C': {'L': 170., 'R': 170.},
                          'L': {'L': 700., 'R': 700.}},
             'NOTO': {'C': {'L': 260., 'R': 260.},
                      'L': {'L': 784., 'R': 784.}},
             'HARTRAO': {'C': {'L': 650., 'R': 650.},
                         'L': {'L': 430., 'R': 430.}},
             'HOBART26': {'C': {'L': 640., 'R': 640.},
                          'L': {'L': 470., 'R': 470.}},
             'MOPRA': {'C': {'L': 350., 'R': 350.},
                       'L': {'L': 340., 'R': 340.},
                       'K': {'L': 900., 'R': 900.}},
             'WARK12M': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'TIDBIN64': {'C': {'L': None, 'R': None},
                          'L': {'L': None, 'R': None}},
             'DSS63': {'C': {'L': 24., 'R': 24.},
                       'L': {'L': 24., 'R': 24.}},
             'PARKES': {'C': {'L': 110., 'R': 110.},
                        'L': {'L': 40., 'R': 40.},
                        'K': {'L': 810., 'R': 810.}},
             'USUDA64': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'JODRELL2': {'C': {'L': 320., 'R': 320.},
                          'L': {'L': 320., 'R': 320.}},
             'ATCA104': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}}}


dtype_converter_dict = {'integer': 'int', 'smallint': 'int', 'character': '|S',
                        'character varying': '|S', 'real': '<f8',
                        'timestamp without time zone': np.object}


def ed_to_uv(r, lambda_cm=18.):
    return r * 12742. * 100000. / lambda_cm


def uv_to_ed(u, lambda_cm=18.):
    return u * lambda_cm / (12742. * 100000)



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
                    band = %(band)s', \
                   {'source': source, 'band': band})
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
        SEFD_rt1 = SEFD_dict[rt1][band.upper()][polar[0]]
    except KeyError:
        print "There's no entry for " + row['st1'] + " for band " +\
              band.upper() + " in utils.SEFD_dict!"
        return None
    except TypeError:
        print "There's no SEFD data for " + row['exper_name'] + " " + \
              row['st1'] + " for band " + band.upper() + " !"
        return None
    try:
        SEFD_rt2 = SEFD_dict[rt2][band.upper()][polar[1]]
    except KeyError:
        print "There's no entry for " + row['st2'] + " for band " + \
              band.upper() + " in utils.SEFD_dict!"
        return None
    except TypeError:
        print "There's no SEFD data for " + row['exper_name'] + " " + \
              row['st2'] + " for band " + band.upper() + " !"
        return None

    try:
        result = (1. / n_q) * math.sqrt((SEFD_rt1 * SEFD_rt2) / (n * dnu *
                                                                 row['solint']))
    except TypeError:
        return None

    return result


def gauss_1d(p, x):
    """
    :param p:
        Parameter vector (amplitude, major axis).
    :param x:
        Numpy array of x-coordinates.
    :return:
        Numpy array of value(s) of gaussian at point(s) (x).
    """
    return p[0] * np.exp(-x ** 2. / (2. * p[1] ** 2.))


def gauss_2d_isotropic(p, x, y):
    """
    :param p:
        Parameter vector (amplitude, major axis).
    :param x:
        Numpy array of x-coordinates.
    :return:
        Numpy array of value(s) of gaussian at point(s) (x, y).
    """
    return p[0] * np.exp(-(x ** 2. + y ** 2.) ** 2. / (2. * p[1] ** 2.))


def gauss_2d_anisotropic(p, x, y):
    """
    :param p:
        Parameter vector (amplitude, major axis, e, rotation angle [from x to
        y]).
    :param x:
        Numpy array of x-coordinates.
    :param y:
        Numpy array of y-coordinates.
    :return:
        Numpy array of value(s) of gaussian at point(s) (x, y).
    """
    a = math.cos(p[3]) ** 2. / (2. * p[1] ** 2.) + math.sin(p[3]) ** 2. /\
                                                   (2. * (p[1] * p[2]) ** 2.)
    b = -math.sin(2. * p[3]) / (4. * p[1] ** 2.) + math.sin(2. * p[3]) /\
                                                   (4. * (p[1] * p[2]) ** 2.)
    c = math.sin(p[3]) ** 2. / (2. * p[1] ** 2.) + math.cos(p[3]) ** 2. / \
                                                   (2. * (p[1] * p[2]) ** 2.)
    return p[0] * np.exp(-(a * x ** 2. + 2. * b * x * y + c * y ** 2.))
