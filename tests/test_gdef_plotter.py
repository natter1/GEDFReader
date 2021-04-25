"""
This file contains tests for gdef_plotter.py.
@author: Nathanael Jöhrmann
"""
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from afm_tools.gdef_sticher import GDEFSticher
from gdef_reader.gdef_importer import GDEFImporter
from gdef_reporter.gdef_plotter import GDEFPlotter

ORIGINAL_FIGURE_SIZE = (4, 3.5)
ORIGINAL_DPI = 300
AUTO_SHOW = True


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
def gdef_measurements():
    example_01_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")
    importer = GDEFImporter(example_01_path)
    gdef_measurements = importer.export_measurements()
    yield gdef_measurements

@pytest.fixture(scope='session')
def gdef_measurements():
    example_01_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")
    importer = GDEFImporter(example_01_path)
    gdef_measurements = importer.export_measurements()
    yield gdef_measurements


@pytest.fixture(scope='session')
def gdef_sticher(gdef_measurement):
    sticher = GDEFSticher([gdef_measurement, gdef_measurement])
    yield sticher


@pytest.fixture(scope="function", params=["GDEFMeasurement", "random_ndarray", "GDEFSticher"])
def data_test_cases(request, gdef_measurement, random_ndarray2d_data, gdef_sticher):
    case_dict = {
        "GDEFMeasurement": gdef_measurement,
        "random_ndarray": random_ndarray2d_data,
        "GDEFSticher": gdef_sticher
    }
    yield case_dict[request.param]


@pytest.fixture(scope='function')
def data_dict(gdef_measurement, random_ndarray2d_data, gdef_sticher):
    data_dict = {
        "GDEFMeasurement": gdef_measurement,
        "random_ndarray": random_ndarray2d_data,
        "GDEFSticher": gdef_sticher
    }
    yield data_dict


@pytest.fixture(scope="function",
                params=["empty", "single ndarray", "single gdef_sticher", "GDEFMeasurements", "mixed dict"])
def data_test_cases_multiple(request, random_ndarray2d_data, gdef_sticher, gdef_measurements, data_dict):
    case_dict = {
        "empty": [],
        "single ndarray": random_ndarray2d_data,
        "single gdef_sticher": gdef_sticher,
        "GDEFMeasurements": gdef_measurements,
        "mixed dict": data_dict
    }
    yield case_dict[request.param]


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
        assert type(fig2) is Figure

    def test_create_rms_per_column_plot(self, gdef_plotter, data_test_cases):
        fig = gdef_plotter.create_rms_per_column_plot(data_test_cases, 1e-6)
        assert type(fig) is Figure

        fig = gdef_plotter.create_rms_per_column_plot(data_test_cases, 1e-6,
                                                      title="create_rms_per_column_figure with title")
        assert fig._suptitle.get_text() == "create_rms_per_column_figure with title\nmoving average n=1 (1.0 µm)"

    def test_create_rms_per_column_plot__multiple(self, gdef_plotter, gdef_measurements, data_dict):
        fig1 = gdef_plotter.create_rms_per_column_plot(gdef_measurements, title="List of GDEFMeasurement")
        assert type(fig1) is Figure

        # Note: all 3 variants of pixel_width should lead to the same result
        # pixel_width = [None, 0.5e-6, None]
        # pixel_width = [10, 0.5e-6, 10]
        pixel_width = 0.5e-6
        fig2 = gdef_plotter.create_rms_per_column_plot(data_dict, pixel_width=pixel_width, title="Dict of DataObject")
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

    def test_create_stich_summary_figure(self, data_test_cases_multiple, gdef_plotter, gdef_measurement, data_dict):
        # sticher_dict = {
        #     "example01": GDEFSticher([gdef_measurement]),# gdef_measurement, gdef_measurement]),
        #     #"example02": GDEFSticher([gdef_measurement]),# gdef_measurement]),
        #     # "example03": GDEFSticher([gdef_measurement]),
        #     # "example04": GDEFSticher([gdef_measurement]),
        #     # "example05": GDEFSticher([gdef_measurement]),
        #     # "example06": GDEFSticher([gdef_measurement]),
        #     # "example07": GDEFSticher([gdef_measurement]),
        #     # "example08": GDEFSticher([gdef_measurement]),
        #     # "example09": GDEFSticher([gdef_measurement]),
        #     # "example10": GDEFSticher([gdef_measurement]),
        #     # "example11": GDEFSticher([gdef_measurement]),
        #     # "example12": GDEFSticher([gdef_measurement]),
        #     # "example13": GDEFSticher([gdef_measurement]),
        #     # "example14": GDEFSticher([gdef_measurement]),
        #     # "example15": GDEFSticher([gdef_measurement]),
        #     # "example16": GDEFSticher([gdef_measurement]),
        #     # "example17": GDEFSticher([gdef_measurement]),
        #     # "example18": GDEFSticher([gdef_measurement]),
        #     # "example19": GDEFSticher([gdef_measurement]),
        #     # "example20": GDEFSticher([gdef_measurement])
        # }
        # fig = gdef_plotter.create_stich_summary_plot(sticher_dict)
        fig = gdef_plotter.create_stich_summary_plot(data_test_cases_multiple)
        #fig = gdef_plotter.create_stich_summary_plot(data_dict)

    def test_create_plot_from_sticher(self, gdef_plotter, gdef_measurement):
        sticher = GDEFSticher([gdef_measurement, gdef_measurement])
        fig = gdef_plotter.create_plot(sticher)
        assert type(fig) is Figure

    # def test_plot_sticher_to_axes(self, gdef_measurement):
    #     sticher = GDEFSticher([gdef_measurement, gdef_measurement])
    #     fig, ax = plt.subplots(figsize=(4, 3), dpi=96)
    #     GDEFPlotter.plot_sticher_to_axes(sticher, ax)
    #     fig.show()