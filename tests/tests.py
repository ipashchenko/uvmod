#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
from unittest import (TestCase, skip, skipIf)
from uvmod.ra_uvfit import (Model_1d, LnLike, LS_estimates, LnPrior,
                      LnPost, hdi_of_mcmc, Model_2d_isotropic,
                      Model_2d_anisotropic)
try:
    from scipy.stats import uniform
    is_scipy = True
except ImportError:
    is_scipy = False
try:
    import emcee
    is_emcee = True
except ImportError:
    is_emcee = False
import numpy as np
import math


# TODO: Add tests for data wo uncertainties
# TODO: Add tests for not installed packages
# TODO: Fix random state to guarantee passing
class Test_1D(TestCase):
    def setUp(self):
        self.p = [2, 0.3]
        self.x = np.array([0., 0.1, 0.2, 0.4, 0.6])
        self.model_1d = Model_1d
        self.model_1d_detections = Model_1d(self.x)
        self.y = self.model_1d_detections(self.p) + np.random.normal(0, 0.1,
                                                                     size=5)
        self.sy = np.random.normal(0.15, 0.025, size=5)
        self.xl = np.array([0.5, 0.7])
        self.yl = np.array([0.6, 0.2])
        self.syl = np.random.normal(0.1, 0.03, size=2)
        self.p1 = np.asarray(self.p) + np.array([1., 0.])
        self.p2 = np.asarray(self.p) + np.array([-1., 0.])
        self.p3 = np.asarray(self.p) + np.array([0., 0.2])
        self.p4 = np.asarray(self.p) + np.array([0., -0.2])
        self.p0_range = [0., 10.]
        self.p1_range = [0., 2.]

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnLike(self):
        lnlike = LnLike(self.x, self.y, self.model_1d, sy=self.sy,
                        x_limits=self.xl, y_limits=self.yl, sy_limits=self.syl,
                        jitter=False, outliers=False)
        lnlik0 = lnlike._lnprob[0].__call__(self.p)
        lnlik1 = lnlike._lnprob[1].__call__(self.p)
        self.assertEqual(lnlike(self.p), lnlik0 + lnlik1)
        self.assertGreater(lnlike(self.p), lnlike(self.p1))
        self.assertGreater(lnlike(self.p), lnlike(self.p2))
        self.assertGreater(lnlike(self.p), lnlike(self.p3))
        self.assertGreater(lnlike(self.p), lnlike(self.p4))

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LS_estimates(self):
        lsq = LS_estimates(self.x, self.y, self.model_1d, sy=self.sy)
        p, pcov = lsq.fit([1., 1.])
        delta0 = 3. * np.sqrt(pcov[0, 0])
        delta1 = 5. * np.sqrt(pcov[1, 1])
        self.assertAlmostEqual(self.p[0], p[0], delta=delta0)
        self.assertAlmostEqual(self.p[1], abs(p[1]), delta=delta1)

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnPrior(self):
        lnprs = ((uniform.logpdf, self.p0_range, dict(),),
                 (uniform.logpdf, self.p1_range, dict(),),)
        lnpr = LnPrior(lnprs)
        self.assertTrue(np.isinf(lnpr([-1., 1.])))
        self.assertTrue(np.isinf(lnpr([1., -1.])))
        self.assertTrue(np.isinf(lnpr([15., 1.])))
        self.assertTrue(np.isinf(lnpr([1., 5.])))

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnPost(self):
        lnprs = ((uniform.logpdf, self.p0_range, dict(),),
                 (uniform.logpdf, self.p1_range, dict(),),)
        lnpr = LnPrior(lnprs)
        lnlike = LnLike(self.x, self.y, self.model_1d, sy=self.sy,
                        x_limits=self.xl, y_limits=self.yl, sy_limits=self.syl,
                        jitter=False, outliers=False)
        lnpost = LnPost(self.x, self.y, self.model_1d, sy=self.sy,
                        x_limits=self.xl, y_limits=self.yl, sy_limits=self.syl,
                        lnpr=lnpr, jitter=False, outliers=False)
        self.assertEqual(lnpost._lnpr(self.p), lnpr(self.p))
        self.assertEqual(lnpost._lnlike(self.p), lnlike(self.p))
        self.assertGreater(lnpost(self.p), lnpost(self.p1))
        self.assertGreater(lnpost(self.p), lnpost(self.p2))
        self.assertGreater(lnpost(self.p), lnpost(self.p3))
        self.assertGreater(lnpost(self.p), lnpost(self.p4))

    @skipIf((not is_emcee) or (not is_scipy), "``emcee`` and/or ``scipy``  not"
                                              " installed")
    def test_MCMC(self):
        nwalkers = 250
        ndim = 2
        p0 = np.random.uniform(low=self.p1_range[0], high=self.p1_range[1],
                               size=(nwalkers, ndim))
        lnprs = ((uniform.logpdf, self.p0_range, dict(),),
                 (uniform.logpdf, self.p1_range, dict(),),)
        lnpr = LnPrior(lnprs)
        lnpost = LnPost(self.x, self.y, self.model_1d, sy=self.sy,
                        x_limits=self.xl, y_limits=self.yl, sy_limits=self.syl,
                        lnpr=lnpr, jitter=False, outliers=False)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
        pos, prob, state = sampler.run_mcmc(p0, 250)
        sampler.reset()
        sampler.run_mcmc(pos, 500)

        sample_vec0 = sampler.flatchain[::10, 0]
        sample_vec1 = sampler.flatchain[::10, 1]
        p0_hdi_min, p0_hdi_max = hdi_of_mcmc(sample_vec0)
        p1_hdi_min, p1_hdi_max = hdi_of_mcmc(sample_vec1)

        self.assertTrue((p0_hdi_min < self.p[0] < p0_hdi_max))
        self.assertTrue((p1_hdi_min < self.p[1] < p1_hdi_max))


class Test_2D_isoptopic(TestCase):
    def setUp(self):
        self.p = [2, 0.3]
        self.x1 = np.random.uniform(low=-1, high=1, size=10)
        self.x2 = np.random.uniform(low=-1, high=1, size=10)
        # Flux at zero uv-spacing - but we need general perfomance
        #self.x[0] = 0.
        #self.y[0] = 0.
        self.xx = np.column_stack((self.x1, self.x2))
        self.model_2d = Model_2d_isotropic
        self.model_2d_detections = Model_2d_isotropic(self.xx)
        self.y = self.model_2d_detections(self.p) + np.random.normal(0, 0.1,
                                                                     size=10)
        self.sy = np.random.normal(0.15, 0.025, size=10)
        self.x1l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                             np.random.uniform(low=0.5, high=1, size=2),))
        self.x2l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                             np.random.uniform(low=0.5, high=1, size=2),))
        self.xxl = np.column_stack((self.x1l, self.x2l))
        self.model_2d_limits = Model_2d_isotropic(self.xxl)
        self.yl = self.model_2d_limits(self.p) + abs(np.random.normal(0, 0.1,
                                                                      size=4))
        self.syl = np.random.normal(0.1, 0.03, size=4)
        self.p1 = np.asarray(self.p) + np.array([1., 0.])
        self.p2 = np.asarray(self.p) + np.array([-1., 0.])
        self.p3 = np.asarray(self.p) + np.array([0., 0.2])
        self.p4 = np.asarray(self.p) + np.array([0., -0.2])
        self.p0_range = [0., 10.]
        self.p1_range = [0., 2.]

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnLike(self):
        lnlike = LnLike(self.xx, self.y, self.model_2d, sy=self.sy,
                        x_limits=self.xxl, y_limits=self.yl, sy_limits=self.syl,
                        jitter=False, outliers=False)
        lnlik0 = lnlike._lnprob[0].__call__(self.p)
        lnlik1 = lnlike._lnprob[1].__call__(self.p)
        self.assertEqual(lnlike(self.p), lnlik0 + lnlik1)
        self.assertGreater(lnlike(self.p), lnlike(self.p1))
        self.assertGreater(lnlike(self.p), lnlike(self.p2))
        self.assertGreater(lnlike(self.p), lnlike(self.p3))
        self.assertGreater(lnlike(self.p), lnlike(self.p4))

    #@skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LS_estimates(self):
        lsq = LS_estimates(self.xx, self.y, self.model_2d, sy=self.sy)
        p, pcov = lsq.fit([1., 1.])
        delta0 = 3. * np.sqrt(pcov[0, 0])
        delta1 = 5. * np.sqrt(pcov[1, 1])
        self.assertAlmostEqual(self.p[0], p[0], delta=delta0)
        # FIXME: use variance as parameter so p[1] > 0
        self.assertAlmostEqual(self.p[1], abs(p[1]), delta=delta1)

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnPost(self):
        lnprs = ((uniform.logpdf, self.p0_range, dict(),),
                 (uniform.logpdf, self.p1_range, dict(),),)
        lnpr = LnPrior(lnprs)
        lnlike = LnLike(self.xx, self.y, self.model_2d, sy=self.sy,
                        x_limits=self.xxl, y_limits=self.yl, sy_limits=self.syl,
                        jitter=False, outliers=False)
        lnpost = LnPost(self.xx, self.y, self.model_2d, sy=self.sy,
                        x_limits=self.xxl, y_limits=self.yl, sy_limits=self.syl,
                        lnpr=lnpr, jitter=False, outliers=False)
        self.assertEqual(lnpost._lnpr(self.p), lnpr(self.p))
        self.assertEqual(lnpost._lnlike(self.p), lnlike(self.p))
        self.assertGreater(lnpost(self.p), lnpost(self.p1))
        self.assertGreater(lnpost(self.p), lnpost(self.p2))
        self.assertGreater(lnpost(self.p), lnpost(self.p3))
        self.assertGreater(lnpost(self.p), lnpost(self.p4))

    @skipIf((not is_emcee) or (not is_scipy), "``emcee`` and/or ``scipy``  not"
                                              " installed")
    def test_MCMC(self):
        nwalkers = 250
        ndim = 2
        p0 = np.random.uniform(low=self.p1_range[0], high=self.p1_range[1],
                               size=(nwalkers, ndim))
        lnprs = ((uniform.logpdf, self.p0_range, dict(),),
                 (uniform.logpdf, self.p1_range, dict(),),)
        lnpr = LnPrior(lnprs)
        lnpost = LnPost(self.xx, self.y, self.model_2d, sy=self.sy,
                        x_limits=self.xxl, y_limits=self.yl, sy_limits=self.syl,
                        lnpr=lnpr, jitter=False, outliers=False)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
        pos, prob, state = sampler.run_mcmc(p0, 250)
        sampler.reset()
        sampler.run_mcmc(pos, 500)

        sample_vec0 = sampler.flatchain[::10, 0]
        sample_vec1 = sampler.flatchain[::10, 1]
        p0_hdi_min, p0_hdi_max = hdi_of_mcmc(sample_vec0)
        p1_hdi_min, p1_hdi_max = hdi_of_mcmc(sample_vec1)

        self.assertTrue((p0_hdi_min < self.p[0] < p0_hdi_max))
        self.assertTrue((p1_hdi_min < self.p[1] < p1_hdi_max))


@skip
class Test_2D_anisoptopic(TestCase):
    def setUp(self):
        self.p = [2, 0.3, 0.7, 0.]
        self.x1 = np.random.uniform(low=-1, high=1, size=10)
        self.x2 = np.random.uniform(low=-1, high=1, size=10)
        # Flux at zero uv-spacing - but we need general perfomance
        #self.x[0] = 0.
        #self.y[0] = 0.
        self.xx = np.column_stack((self.x1, self.x2))
        self.model_2d_anisotropic = Model_2d_anisotropic
        self.model_2d_detections = Model_2d_anisotropic(self.xx)
        self.y = self.model_2d_detections(self.p) + np.random.normal(0, 0.05,
                                                                     size=10)
        self.sy = np.random.normal(0.15, 0.025, size=10)
        self.x1l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                              np.random.uniform(low=0.5, high=1, size=2),))
        self.x2l = np.hstack((np.random.uniform(low=-1, high=-0.5, size=2),
                              np.random.uniform(low=0.5, high=1, size=2),))
        self.xxl = np.column_stack((self.x1l, self.x2l))
        self.model_2d_limits = Model_2d_anisotropic(self.xxl)
        self.yl = self.model_2d_limits(self.p) + abs(np.random.normal(0, 0.05,
                                                                      size=4))
        self.syl = np.random.normal(0.1, 0.03, size=4)
        self.p1 = np.asarray(self.p) + np.array([1., 0., 0., 0.])
        self.p2 = np.asarray(self.p) + np.array([-1., 0., 0., 0.])
        self.p3 = np.asarray(self.p) + np.array([0., 0.2, 0., 0.])
        self.p4 = np.asarray(self.p) + np.array([0., -0.2, 0., 0.])
        self.p5 = np.asarray(self.p) + np.array([0., 0., 0.4, 0.])
        self.p6 = np.asarray(self.p) + np.array([0., 0., -0.4, 0.])
        self.p7 = np.asarray(self.p) + np.array([0., 0., 0., math.pi / 2.])
        self.p8 = np.asarray(self.p) + np.array([0., 0., 0., -math.pi / 2.])
        self.p0_range = [0., 10.]
        self.p1_range = [0., 2.]
        self.p2_range = [0., 2.]
        self.p3_range = [0., math.pi]

    @skipIf(not is_scipy, "``scipy`` is not installed")
    def test_LnLike(self):
        lnlike = LnLike(self.xx, self.y, self.model_2d_anisotropic, sy=self.sy,
                        x_limits=self.xxl, y_limits=self.yl, sy_limits=self.syl,
                        jitter=False, outliers=False)
        lnlik0 = lnlike._lnprob[0].__call__(self.p)
        lnlik1 = lnlike._lnprob[1].__call__(self.p)
        self.assertEqual(lnlike(self.p), lnlik0 + lnlik1)
        self.assertGreater(lnlike(self.p), lnlike(self.p1))
        self.assertGreater(lnlike(self.p), lnlike(self.p2))
        self.assertGreater(lnlike(self.p), lnlike(self.p3))
        self.assertGreater(lnlike(self.p), lnlike(self.p4))
        self.assertGreater(lnlike(self.p), lnlike(self.p5))
        self.assertGreater(lnlike(self.p), lnlike(self.p6))
        self.assertGreater(lnlike(self.p), lnlike(self.p7))
        self.assertGreater(lnlike(self.p), lnlike(self.p8))
