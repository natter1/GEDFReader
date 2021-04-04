"""
This file contains tests for gdef_plotter.py.
@author: Nathanael Jöhrmann
"""
import pytest

from gdef_reporter.gdef_plotter import GDEFPlotter


@pytest.fixture(scope='session')
def gdef_plotter():
    gdef_plotter = GDEFPlotter(figure_size=(12, 9), dpi=300)
    yield gdef_plotter

class TestGDEFPlotter:
    # properties
    def test_dpi(self, gdef_plotter):
        assert gdef_plotter.dpi == 300
        old_dpi = gdef_plotter.dpi
        gdef_plotter.dpi = 200
        assert gdef_plotter.dpi == 200
        gdef_plotter.dpi = old_dpi
        assert gdef_plotter.dpi == 300

    def test_figure_size(self, gdef_plotter):
        assert gdef_plotter.figure_size == (12, 9)
        gdef_plotter.figure_size = (4, 3)
        assert gdef_plotter.figure_size == (4, 3)