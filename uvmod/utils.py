import math
import numpy as np


band_cm_dict = {'c': 6., 'l': 18., 'p': 94., 'k': 1.35 }

def ed_to_uv(r, lambda_cm=18.):
    return r * 12742. * 100000. / lambda_cm


def uv_to_ed(u, lambda_cm=18.):
    return u * lambda_cm / (12742. * 100000)


vfloat = np.vectorize(float)
n_q = 0.637
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
