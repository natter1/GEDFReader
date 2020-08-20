"""
This file contains tests for gdef_measurement.py.
@author: Nathanael Jöhrmann
"""
import pytest

from gdef_reader.gdef_measurement import GDEFSettings


@pytest.fixture(scope='session')
def gdef_setting():
    gdef_setting = GDEFSettings()
    yield gdef_setting


class TestGDEFSettings:
    def test_pixel_width(self, pptx_creator):
        assert False

    def test_pixel_area(self, pptx_creator):
        assert False

    def test_size_in_um_for_plot(self, pptx_creator):
        assert False

    def test_shape(self, pptx_creator):
        assert False
