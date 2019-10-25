import torch
import pytest
import math

import deepsmlm.generic.phot_camera as pc
import deepsmlm.test.utils_ci as tutil


class TestPhotons2Camera:

    @pytest.fixture(scope='class')
    def m2_spec(self):
        return pc.Photon2Camera(qe=1.0, spur_noise=0.002, em_gain=300., e_per_adu=45.,
                                baseline=100, read_sigma=74.4, photon_units=False)

    def test_shape(self, m2_spec):
        x = torch.ones((32, 3, 64, 64))
        assert tutil.tens_eqshape(x, m2_spec.forward(x))

    def test_photon_units(self, m2_spec):
        m2_spec.photon_units = True

        x = torch.rand((32, 3, 64, 64)) * 2000
        out = m2_spec.forward(x)

        tol = 0.01
        assert abs((x.mean() - out.mean()) / x.mean()) <= tol

    def test_warning(self, m2_spec):
        m2_spec.photon_units = True
        x = torch.rand((32, 3, 64, 64))
        out = m2_spec.reverse(x)



