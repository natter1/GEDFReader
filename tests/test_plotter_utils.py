"""
This file contains tests for plotter_utils.py.
@author: Nathanael Jöhrmann
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from afm_tools.gdef_sticher import GDEFSticher
from gdef_reader.gdef_importer import GDEFImporter
from gdef_reporter.plotter_utils import plot_to_ax, create_plot, plot_z_histogram_to_ax, create_z_histogram_plot, \
    _extract_ndarray_and_pixel_width, save_figure, create_rms_plot

ORIGINAL_FIGURE_SIZE = (4, 3.5)
ORIGINAL_DPI = 300
AUTO_SHOW = True


def auto_show(fig):
    if AUTO_SHOW:
        fig.show()


@pytest.fixture(scope='session')
def random_ndarray2d_data():
    # np.random.seed(1)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    yield rs.random((256, 256)) * 1e-6 - 0.5e-6


@pytest.fixture(scope='session')
def gdef_measurement():
    example_01_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")
    importer = GDEFImporter(example_01_path)
    gdef_measurement = importer.export_measurements()[0]
    yield gdef_measurement


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


# @pytest.fixture()
# def figure_and_ax():
#     fig, ax = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
#     yield fig, ax


# tests for functions to plot a 2D area map
class TestAreaPlots:
    def test_plot_to_ax(self, data_test_cases):
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        plot_to_ax(ax1, data_test_cases, pixel_width=1.0)
        auto_show(fig1)
        assert type(fig1) is Figure
        assert fig1.axes[1].get_title() == "nm"  # default unit for z-values should be nm

        fig2, ax2 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        pixel_width = 5.0
        title = f"{type(data_test_cases).__name__}\npixel_width={pixel_width}"
        plot_to_ax(ax2, data_test_cases, pixel_width=pixel_width, title=title)
        auto_show(fig2)
        assert ax2.get_title() == title

        fig3, ax3 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        z_factor = 1.0
        title = f"{type(data_test_cases).__name__}\nz_unit: [m] - z_factor={z_factor}"
        plot_to_ax(ax3, data_test_cases, pixel_width=5.0, z_unit="µm", title=title)
        auto_show(fig3)
        assert fig3.axes[1].get_title() == "\u03BCm"

    def test_create_plot(self, data_test_cases):
        fig1 = create_plot(data_test_cases, 1e-6, "default value for cropped (True)", ORIGINAL_FIGURE_SIZE,
                           ORIGINAL_DPI)
        auto_show(fig1)
        assert np.any(comparison := (fig1.get_size_inches() < ORIGINAL_FIGURE_SIZE)) and not np.all(comparison)
        assert fig1.dpi == ORIGINAL_DPI

        fig2 = create_plot(data_test_cases, 1e-6, "cropped=False", max_figure_size=ORIGINAL_FIGURE_SIZE, cropped=False)
        assert np.all(fig2.get_size_inches() == ORIGINAL_FIGURE_SIZE)
        auto_show(fig2)


class Test1DPlots:
    def test_plot_z_histogram_to_ax__defaults(self, data_test_cases):
        # first, check default behaviour of parameters title, , n_bins, units and add_norm
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        plot_z_histogram_to_ax(ax1, data_test_cases, title="")
        auto_show(fig1)
        assert len(ax1.lines) == 0  # no Gauss fit (expected default behaviour)
        assert ax1.get_title().startswith("\u03BC=")  # default title starts with mu=...
        assert ax1.get_xlabel() == "z [\u03BCm]"  # default units should be µm; note:  µ == \u03BC is False!
        assert len(ax1.containers[0]) == 200  # default n_bins should be 200

    def test_plot_z_histogram_to_ax__set_parameters(self, data_test_cases):
        # first, check setting a title, selecting units µm, set n_bins and draw normal distribution fit
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        title = "Use [µm] and Gauss fit"
        n_bins = 20
        plot_z_histogram_to_ax(ax1, data_test_cases, n_bins=n_bins, units="nm", title=title, add_norm=True)
        auto_show(fig1)
        assert len(ax1.lines) == 1  # Gauss fit (add_norm=True)
        assert ax1.get_title() == title
        assert str(ax1.get_xlabel()) == str(f"z [nm]")  # note: comparison between µ and \u03BC is False!
        assert len(ax1.containers[0]) == n_bins  # default n_bins should be 200

        # second, check no title via title=None
        fig2, ax2 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        title = None
        plot_z_histogram_to_ax(ax2, data_test_cases, n_bins=20, units="µm", title=title)
        auto_show(fig2)
        assert ax2.get_title() == ""  # expected for title=None

    def test_plot_z_histogram_to_ax__multiple_datas(self, gdef_measurement, random_ndarray2d_data):
        data_list = [gdef_measurement, random_ndarray2d_data]
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        plot_z_histogram_to_ax(ax1, data_list, title="", add_norm=True)
        auto_show(fig1)
        assert len(fig1.axes[0].containers) == len(data_list)
        assert len(fig1.axes[0].lines) == 2  # Gauss fits (add_norm=True)

    def test_create_z_histogram_plot__defaults(self, data_test_cases):
        # check setting figure_size and dpi and also default of parameters title, n_bins, units and add_norm
        fig1 = create_z_histogram_plot(data_test_cases, figure_size=ORIGINAL_FIGURE_SIZE, dpi=ORIGINAL_DPI)
        auto_show(fig1)
        assert type(fig1) is Figure
        assert np.any(fig1.get_size_inches() == ORIGINAL_FIGURE_SIZE)
        assert fig1.dpi == ORIGINAL_DPI
        assert len(fig1.axes[0].lines) == 0  # no Gauss fit (expected default behaviour)
        assert fig1.axes[0].get_title().startswith("\u03BC=")  # default title starts with mu=...
        assert fig1.axes[0].get_xlabel() == "z [\u03BCm]"  # default units should be µm; note:  µ == \u03BC is False!
        assert len(fig1.axes[0].containers[0]) == 200  # default n_bins should be 200

    def test_create_z_histogram_plot__set_paramaters(self, data_test_cases):
        # first, check setting label, a title, selecting units µm, set n_bins and draw normal distribution fit
        labels = type(data_test_cases).__name__
        title = "Use [nm] and Gauss fit"
        n_bins = 20
        fig1 = create_z_histogram_plot(data_test_cases, labels, n_bins=n_bins, title=title, units="nm",
                                       add_norm=True, figure_size=ORIGINAL_FIGURE_SIZE, dpi=ORIGINAL_DPI)
        auto_show(fig1)
        assert len(fig1.axes[0].lines) == 1  # Gauss fit (add_norm=True)
        assert fig1._suptitle.get_text() == title
        assert fig1.axes[0].get_title() == ""
        assert str(fig1.axes[0].get_xlabel()) == str(f"z [nm]")  # note: comparison between µ and \u03BC is False!
        assert len(fig1.axes[0].containers[0]) == n_bins  # default n_bins should be 200

        # second, check no title via title=None
        fig2 = create_z_histogram_plot(data_test_cases, title=None, figure_size=ORIGINAL_FIGURE_SIZE, dpi=ORIGINAL_DPI)
        auto_show(fig2)
        assert fig2._suptitle is None
        assert fig2.axes[0].get_title() == ""

    def test_create_z_histogram_plot__multiple_datas(self, gdef_measurement, random_ndarray2d_data):
        data_list = [gdef_measurement, random_ndarray2d_data]
        labels = []
        for i, data in enumerate(data_list):
            labels.append(f"{i} - {type(data).__name__}")
        fig1 = create_z_histogram_plot(data_list, labels=labels, title="", add_norm=True)
        auto_show(fig1)
        assert len(fig1.axes[0].containers) == len(data_list)
        assert len(fig1.axes[0].lines) == 2  # Gauss fits (add_norm=True)

    def test_plot_rms_to_ax(self):
        pass

    def test_create_rms_plot(self, gdef_measurement, random_ndarray2d_data, gdef_sticher):
        data_list = [gdef_measurement, gdef_sticher, random_ndarray2d_data*0.07]
        pixel_width = gdef_measurement.pixel_width
        labels = [f"{type(data).__name__}" for data in data_list]
        fig = create_rms_plot(data_list, label_list=labels, pixel_width=pixel_width, moving_average_n=1,
                              subtract_average=True, units="nm")
        auto_show(fig)
        fig = create_rms_plot(data_list, label_list=labels, pixel_width=pixel_width, moving_average_n=1,
                              subtract_average=True)
        auto_show(fig)


class TestSpecialFunctions:
    def test_extract_ndarray_and_pixel_width(self, data_test_cases):
        pixel_width = 1
        ndarray2d, px_width = _extract_ndarray_and_pixel_width(data_test_cases, pixel_width=pixel_width)
        assert type(ndarray2d) is np.ndarray
        if isinstance(data_test_cases, np.ndarray):
            assert np.all(data_test_cases == ndarray2d)
            assert px_width == pixel_width
        else:
            assert np.all(data_test_cases.values == ndarray2d)
            assert data_test_cases.pixel_width == px_width

    def test_save_figure(self, tmp_path):
        fig, _ = plt.subplots(1, 1, dpi=72, figsize=(1, 1), constrained_layout=True)

        # first try saving in existing folder with default settings
        assert tmp_path.exists()
        filename = "default"
        save_figure(fig, tmp_path, filename)
        png_file = tmp_path / f"{filename}.png"  # should be saved by default
        pdf_file = tmp_path / f"{filename}.pdf"  # should not be saved by default
        assert png_file.exists()
        assert not pdf_file.exists()

        # second, save nothing:
        filename = "save_nothing"
        save_figure(fig, tmp_path, filename, png=False, pdf=False)
        png_file = tmp_path / f"{filename}.png"  # should be saved by default
        pdf_file = tmp_path / f"{filename}.pdf"  # should not be saved by default
        assert not png_file.exists()
        assert not pdf_file.exists()

        # third, only save pdf
        filename = "save_pdf"
        save_figure(fig, tmp_path, filename, png=False, pdf=True)
        png_file = tmp_path / f"{filename}.png"  # should be saved by default
        pdf_file = tmp_path / f"{filename}.pdf"  # should not be saved by default
        assert not png_file.exists()
        assert pdf_file.exists()

        # fourth, use folder that does not exist jet and save both png and pdf
        new_tmp_path = tmp_path / "new/"
        assert not new_tmp_path.exists()
        filename = "save_pdf_and_png"
        save_figure(fig, new_tmp_path, filename, png=True, pdf=True)
        png_file = new_tmp_path / f"{filename}.png"  # should be saved by default
        pdf_file = new_tmp_path / f"{filename}.pdf"  # should not be saved by default
        assert png_file.exists()
        assert pdf_file.exists()
