from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def scatter_3d(x1, x2, y, xlabel='u', ylabel='v', zlabel='flux', xlim3d=None,
               ylim3d=None, zlim3d=None):
    """
    Do 3d scatter plot.
    :param x1:
        Array-like of first coordinate.
    :param x2:
        Array-like of second coordinate.
    :param y:
        Array-like of values.
    :param xlabel:
    :param ylabel:
    :param zlabel:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='r', marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    #configure axes
    if xlim3d is not None:
        ax.set_xlim3d(xlim3d[0], xlim3d[1])
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d[0], ylim3d[1])
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d[0], zlim3d[1])
    plt.show()


def scatter_3d_errorbars(x1, x2, y, sy, xlabel='u', ylabel='v', zlabel='flux',
                         xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Do 3d plot with errorbars.
    :param x1:
        Array-like of first coordinate.
    :param x2:
        Array-like of second coordinate.
    :param y:
        Array-like of values.
    :param sy:
        Array-like of errors.
    :param xlabel:
    :param ylabel:
    :param zlabel:
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y = np.asarray(y)
    sy = np.asarray(sy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot points
    ax.scatter(x1, x2, y, c='r', marker='o')
    #plot errorbars
    for i in np.arange(0, len(x1)):
        ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [y[i] + sy[i], y[i] - sy[i]],
                marker="_", color='r')
    #configure axes
    if xlim3d is not None:
        ax.set_xlim3d(xlim3d[0], xlim3d[1])
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d[0], ylim3d[1])
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d[0], zlim3d[1])
    plt.show()
