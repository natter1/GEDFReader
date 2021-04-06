"""
This file contains tests for gdef_plotter.py.
@author: Nathanael Jöhrmann
"""
import pytest
import numpy as np
from gdef_reporter.gdef_plotter import GDEFPlotter
from matplotlib.figure import Figure

@pytest.fixture(scope="function", params=[-4, 0, 0.4, 1, 2])
def _test_case(request):
    yield request.param


@pytest.fixture(scope='session')
def random_ndarray2d_data():
    # np.random.seed(1)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    yield rs.random((256, 256))


@pytest.fixture(scope='session')
def gdef_plotter():
    gdef_plotter = GDEFPlotter(figure_size=(12, 9), dpi=300, auto_show=True)
    yield gdef_plotter


class TestGDEFPlotter:
    # properties
    def test_dpi(self, gdef_plotter):
        assert gdef_plotter.dpi == 300
        old_dpi = gdef_plotter.dpi
        # change dpi manually
        gdef_plotter.dpi = 200
        assert gdef_plotter.dpi == 200
        assert gdef_plotter.plotter_style_rms.dpi == 200
        assert gdef_plotter.plotter_style_sigma.dpi == 200
        # combine reading and writing dpi
        gdef_plotter.dpi = old_dpi
        assert gdef_plotter.dpi == 300

    def test_figure_size(self, gdef_plotter):
        assert gdef_plotter.figure_size == (12, 9)
        gdef_plotter.figure_size = (4, 3)
        assert gdef_plotter.figure_size == (4, 3)

    def test_set_dpi_and_figure_size(self, gdef_plotter):
        old_dpi = gdef_plotter.dpi
        old_figure_size = gdef_plotter.figure_size
        gdef_plotter.set_dpi_and_figure_size()
        assert old_dpi == gdef_plotter.dpi
        assert old_figure_size == gdef_plotter.figure_size

        gdef_plotter.set_dpi_and_figure_size(old_dpi+1, (old_figure_size[0]+1, old_figure_size[1]))
        assert old_dpi + 1 == gdef_plotter.dpi
        assert old_figure_size[0]+1, old_figure_size[1] == gdef_plotter.figure_size

        gdef_plotter.set_dpi_and_figure_size(old_dpi, old_figure_size)

    def test_create_surface_figure(self, gdef_plotter, random_ndarray2d_data):
        assert gdef_plotter.create_surface_figure(None, 1e-6) is None
        fig1 = gdef_plotter.create_surface_figure(random_ndarray2d_data, 1e-6)
        fig2 = gdef_plotter.create_surface_figure(random_ndarray2d_data, 1e-6, cropped=False)

        assert type(fig1) is Figure
        assert type(fig2) is Figure
