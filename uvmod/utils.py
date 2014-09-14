import math
import numpy as np


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
