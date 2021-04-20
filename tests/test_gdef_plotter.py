"""
This file contains tests for gdef_plotter.py.
@author: Nathanael Jöhrmann
"""
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from afm_tools.background_correction import BGCorrectionType
from afm_tools.gdef_sticher import GDEFSticher
from gdef_reader.gdef_importer import GDEFImporter
from gdef_reporter.gdef_plotter import GDEFPlotter

ORIGINAL_FIGURE_SIZE = (4, 3.5)
ORIGINAL_DPI = 300
AUTO_SHOW = True


@pytest.fixture(scope="function", params=[-4, 0, 0.4, 1, 2])
def data_test_cases(request):
    yield request.param


@pytest.fixture(scope='session')
def random_ndarray2d_data():
    # np.random.seed(1)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    yield rs.random((256, 256)) * 1e-6


@pytest.fixture(scope='session')
def gdef_measurement():
    example_01_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")
    importer = GDEFImporter(example_01_path)
    gdef_measurement = importer.export_measurements()[0]
    yield gdef_measurement


@pytest.fixture(scope='session')
def gdef_plotter():
    GDEFPlotter.auto_show = AUTO_SHOW
    gdef_plotter = GDEFPlotter(figure_size=ORIGINAL_FIGURE_SIZE, dpi=ORIGINAL_DPI)
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

    def test_create_plot(self, gdef_plotter, random_ndarray2d_data):
        fig1 = gdef_plotter.create_plot(random_ndarray2d_data, 1e-6, title="create_surface_figure")
        fig2 = gdef_plotter.create_plot(random_ndarray2d_data, 1e-6,
                                        title="create_surface_figure with cropped=False", cropped=False)

        assert type(fig1) is Figure
        assert type(fig2) is Figure

    def test_create_rms_per_column_plot(self, gdef_plotter, random_ndarray2d_data):
        fig = gdef_plotter.create_rms_per_column_plot(random_ndarray2d_data, 1e-6)
        assert type(fig) is Figure

        fig = gdef_plotter.create_rms_per_column_plot(random_ndarray2d_data, 1e-6,
                                                      title="create_rms_per_column_figure with title")
        assert fig._suptitle.get_text() == "create_rms_per_column_figure with title\nmoving average n=1 (1.0 µm)"

    def test_create_absolute_gradient_rms_plot(self, gdef_plotter, random_ndarray2d_data):
        cutoff_list = [1, 10, 20, 50, 90, 100]

        fig = gdef_plotter.create_absolute_gradient_rms_plot(
            random_ndarray2d_data, cutoff_list, 1e-6, moving_average_n=10)
        assert type(fig) is Figure
        fig = gdef_plotter.create_absolute_gradient_rms_plot(
            random_ndarray2d_data, cutoff_list, 1e-6, moving_average_n=10, title="create_absolute_gradient_rms_figure")
        assert type(fig) is Figure

    def test_create_sigma_moving_average_plot(self, gdef_plotter, gdef_measurement, random_ndarray2d_data):
        # gdef_measurement.values = random_ndarray2d_data
        # gdef_measurement.correct_background(BGCorrectionType.raw_data)
        sticher_dict = {"example01": GDEFSticher([gdef_measurement])}
        gdef_plotter.create_sigma_moving_average_plot_from_sticher_dict(sticher_dict, 10)
        pass  # needs sticher_dict: Dict[str, GDEFSticher]

    def test_create_absolute_gradient_plot(self, gdef_plotter, random_ndarray2d_data):
        cutoff_list = [1, 10, 20, 50, 90, 100]
        fig = gdef_plotter.create_absolute_gradient_maps_plot(
            random_ndarray2d_data, cutoff_list)

    def test_create_stich_summary_figure(self, gdef_plotter, gdef_measurement):
        sticher_dict = {
            # "example01": GDEFSticher([gdef_measurement, gdef_measurement, gdef_measurement]),
            "example02": GDEFSticher([gdef_measurement, gdef_measurement])
        }
        fig = gdef_plotter.create_stich_summary_plot(sticher_dict)

    def test_create_plot_from_sticher(self, gdef_plotter, gdef_measurement):
        sticher = GDEFSticher([gdef_measurement, gdef_measurement])
        fig = gdef_plotter.create_plot_from_sticher(sticher)
        assert type(fig) is Figure

    # def test_plot_sticher_to_axes(self, gdef_measurement):
    #     sticher = GDEFSticher([gdef_measurement, gdef_measurement])
    #     fig, ax = plt.subplots(figsize=(4, 3), dpi=96)
    #     GDEFPlotter.plot_sticher_to_axes(sticher, ax)
    #     fig.show()