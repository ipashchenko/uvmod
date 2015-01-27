import numpy as np
import os
import sys
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from uvmod import models


if __name__ == "__main__":
    # Create 1D data
    # ``True`` parameters
    p = [2, 0.7]

    # Create predictors
    x = np.random.uniform(low=0, high=1, size=10)

    model_1d = models.Model_1d
    model_1d_detections = models.Model_1d(x)

    y = model_1d_detections(p) + np.random.normal(0, 0.05,size=10)
    # Generate noise estimates (std)
    sy = np.random.normal(0.15, 0.025, size=10)

    # Create upper limits predictors in outer part of x axis
    xl = np.random.uniform(low=0.5, high=1, size=2)
    model_1d_limits = models.Model_1d(xl)
    # Generate upper limits
    yl = model_1d_limits(p) + abs(np.random.normal(0, 0.05, size=2))
    # Generate noise estimates (std)
    syl = np.random.normal(0.1, 0.03, size=2)
    # Save data
    np.savetxt('detections1d.dat', np.concatenate((x[:, None], y[:, None],
                                                   sy[:, None]), axis=1))
    np.savetxt('ulimits1d.dat', np.concatenate((xl[:, None], yl[:, None],
                                                syl[:, None]), axis=1))

    # Create 2D data
    # ``True`` parameters
    p = [2, 0.7, 0.3, 1.]

    # Create predictors
    x1 = np.random.uniform(low=-1, high=1, size=10)
    x2 = np.random.uniform(low=-1, high=1, size=10)
    xx = np.column_stack((x1, x2))

    model_2d_anisotropic = models.Model_2d_anisotropic
    model_2d_detections = models.Model_2d_anisotropic(xx)

    y = model_2d_detections(p) + np.random.normal(0, 0.05,size=10)
    # Generate noise estimates (std)
    sy = np.random.normal(0.15, 0.025, size=10)

    # Create upper limits predictors in outer part of x1-x2 plane
    x1l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                                  np.random.uniform(low=0.5, high=1, size=2),))
    x2l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                                  np.random.uniform(low=0.5, high=1, size=2),))
    xxl = np.column_stack((x1l, x2l))
    model_2d_limits = models.Model_2d_anisotropic(xxl)
    # Generate upper limits
    yl = model_2d_limits(p) + abs(np.random.normal(0, 0.05, size=4))
    # Generate noise estimates (std)
    syl = np.random.normal(0.1, 0.03, size=4)
    # Save data
    np.savetxt('detections2d.dat', np.concatenate((xx, y[:, None], sy[:, None]),
                                                  axis=1))
    np.savetxt('ulimits2d.dat', np.concatenate((xxl, yl[:, None], syl[:, None]),
                                               axis=1))
