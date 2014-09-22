from utils import gauss_1d, gauss_2d_isotropic, gauss_2d_anisotropic


class Model(object):
    """
    Basic class that implements models.

    :param x:
        Numpy array of coordinates with shape(ndim, npoints,).
    """
    def __init__(self, x):
        self.x = x

    def __call__(self, p):
        raise NotImplementedError


class Model_1d(Model):
    def __call__(self, p):
        """
        :param p:
            Array-like parameter vector (amplitude, axis std)
        :return:
            Numpy array of values of the model with parameters represented by
            array-like ``p`` for points, specified in constructor.
        """
        x = self.x
        return gauss_1d(p, x)


class Model_2d_isotropic(Model):
    def __call__(self, p):
        """
        :param p:
            Array-like parameter vector (amplitude, axis std)
        :return:
            Numpy array of values of the model with parameters represented by
            array-like ``p`` for points, specified in constructor.
        """
        x = self.x[:, 0]
        y = self.x[:, 1]
        return gauss_2d_isotropic(p, x, y)


class Model_2d_anisotropic(Model):
    def __call__(self, p):
        """
        :param p:
            Array-like parameter vector (amplitude, major axis std,
            minor-to-major axis ratio, rotation angle)
        :return:
            Numpy array of values of the model with parameters represented by
            array-like ``p`` for points, specified in constructor.
        """
        x = self.x[:, 0]
        y = self.x[:, 1]
        return gauss_2d_anisotropic(p, x, y)
