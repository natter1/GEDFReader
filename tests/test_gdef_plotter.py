"""
This file contains tests for gdef_plotter.py.
@author: Nathanael Jöhrmann
"""

import pytest
from matplotlib.figure import Figure

from afm_tools.gdef_sticher import GDEFSticher
from gdef_reporter.gdef_plotter import GDEFPlotter
from tests.conftest import AUTO_SHOW

ORIGINAL_FIGURE_SIZE = (4, 3.5)
ORIGINAL_DPI = 300


@pytest.fixture(scope='session')
def gdef_plotter():
    gdef_plotter = GDEFPlotter(figure_size=ORIGINAL_FIGURE_SIZE, dpi=ORIGINAL_DPI, auto_show=AUTO_SHOW)
    yield gdef_plotter


class TestGDEFPlotter:
    # properties
    def test_dpi(self, gdef_plotter):
        assert gdef_plotter.dpi == ORIGINAL_DPI
        old_dpi = gdef_plotter.dpi
        # change dpi manually
        gdef_plotter.dpi = 200
        assert gdef_plotter.dpi == 200
        assert gdef_plotter.plotter_style_rms.dpi == 200
        assert gdef_plotter.plotter_style_sigma.dpi == 200
        # use old_dpi to test reading and writing dpi
        gdef_plotter.dpi = old_dpi
        assert gdef_plotter.dpi == ORIGINAL_DPI

    def test_figure_size(self, gdef_plotter):
        assert gdef_plotter.figure_size == ORIGINAL_FIGURE_SIZE
        gdef_plotter.figure_size = (4, 3)
        assert gdef_plotter.figure_size == (4, 3)
        gdef_plotter.figure_size = ORIGINAL_FIGURE_SIZE

    def test_set_dpi_and_figure_size(self, gdef_plotter):
        # use default parameters -> should not change anything
        gdef_plotter.set_dpi_and_figure_size()
        assert gdef_plotter.dpi == ORIGINAL_DPI
        assert gdef_plotter.figure_size == ORIGINAL_FIGURE_SIZE

        gdef_plotter.set_dpi_and_figure_size(ORIGINAL_DPI + 1, (ORIGINAL_FIGURE_SIZE[0] + 1, ORIGINAL_FIGURE_SIZE[1]))
        assert ORIGINAL_DPI + 1 == gdef_plotter.dpi
        assert ORIGINAL_FIGURE_SIZE[0] + 1, ORIGINAL_FIGURE_SIZE[1] == gdef_plotter.figure_size

        gdef_plotter.set_dpi_and_figure_size(ORIGINAL_DPI, ORIGINAL_FIGURE_SIZE)

    def test_create_plot(self, gdef_plotter, data_test_cases):
        fig1 = gdef_plotter.create_plot(data_test_cases, 1e-6, title="create_surface_figure")
        fig2 = gdef_plotter.create_plot(data_test_cases, 1e-6,
                                        title="create_surface_figure with cropped=False", cropped=False)

        assert type(fig1) is Figure
        #assert type(fig2) is Figure

    def test_create_rms_per_column_plot(self, gdef_plotter, data_test_cases):
        fig = gdef_plotter.create_rms_per_column_plot(data_test_cases, 1e-6)
        assert type(fig) is Figure

        fig = gdef_plotter.create_rms_per_column_plot(data_test_cases, 1e-6,
                                                      title="create_rms_per_column_figure with title")
        assert fig._suptitle.get_text() == "create_rms_per_column_figure with title\nmoving average n=1 (1.0 µm)"

    def test_create_rms_per_column_plot__multiple(self, gdef_plotter, gdef_measurements, data_test_cases):
        fig1 = gdef_plotter.create_rms_per_column_plot(gdef_measurements, title="List of GDEFMeasurement")
        assert type(fig1) is Figure

        # Note: all 3 variants of pixel_width should lead to the same result
        # pixel_width = [None, 0.5e-6, None]
        # pixel_width = [10, 0.5e-6, 10]
        pixel_width = 0.5e-6
        fig2 = gdef_plotter.create_rms_per_column_plot(data_test_cases, pixel_width=pixel_width, title="Dict of DataObject")
        assert type(fig2) is Figure

    def test__create_absolute_gradient_rms_plot(self, gdef_plotter, random_ndarray2d_data):
        cutoff_list = [1, 10, 20, 50, 90, 100]

        fig = gdef_plotter._create_absolute_gradient_rms_plot(
            random_ndarray2d_data, cutoff_list, 1e-6, moving_average_n=10)
        assert type(fig) is Figure
        fig = gdef_plotter._create_absolute_gradient_rms_plot(
            random_ndarray2d_data, cutoff_list, 1e-6, moving_average_n=10, title="create_absolute_gradient_rms_figure")
        assert type(fig) is Figure

    def test__create_absolute_gradient_maps_plot(self, gdef_plotter, random_ndarray2d_data):
        cutoff_list = [1, 10, 20, 50, 90, 100]
        fig = gdef_plotter._create_absolute_gradient_maps_plot(
            random_ndarray2d_data, cutoff_list)

    def test_create_stich_summary_plot(self, multiple_data_test_cases, gdef_plotter, gdef_measurement):
        pixel_width=0.5e-6
        fig = gdef_plotter.create_stich_summary_plot(multiple_data_test_cases, pixel_width=pixel_width)
        #fig = gdef_plotter.create_stich_summary_plot(data_dict)

    def test_create_rms_with_error_plot_from_sticher_dict(self, gdef_plotter, gdef_measurement):
        sticher = GDEFSticher([gdef_measurement, gdef_measurement])
        fig = gdef_plotter.create_rms_with_error_plot_from_sticher_dict({"test": sticher})
        assert type(fig) is Figure
