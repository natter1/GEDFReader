"""
This file contains tests for plotter_utils.py.
@author: Nathanael Jöhrmann
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from afm_tools.gdef_sticher import GDEFSticher
from gdef_reporter.plotter_utils import plot_to_ax, create_plot, plot_z_histogram_to_ax, create_z_histogram_plot, \
    _extract_ndarray_and_pixel_width, save_figure, create_rms_plot, create_rms_with_error_plot
from tests.conftest import AUTO_SHOW

ORIGINAL_FIGURE_SIZE = (4, 3.5)
ORIGINAL_DPI = 300


def auto_show(fig):
    if AUTO_SHOW:
        fig.show()


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


class Test1DPlotZHistogram:
    def test_plot_z_histogram_to_ax__defaults(self, data_test_cases):
        # first, check default behaviour of parameters title, , n_bins, units and add_norm
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        plot_z_histogram_to_ax(ax1, data_test_cases, title="")
        auto_show(fig1)
        assert len(ax1.lines) == 0  # no Gauss fit (expected default behaviour)
        assert ax1.get_title().startswith("\u03BC=")  # default title starts with mu=...
        assert ax1.get_xlabel() == "z [\u03BCm]"  # default units should be µm; note:  µ == \u03BC is False!
        assert len(ax1.containers[0]) == 200  # default n_bins should be 200

    def test_plot_z_histogram_to_ax__defaults_multiple(self, multiple_data_test_cases):
        # first, check default behaviour of parameters title, , n_bins, units and add_norm
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)

        if isinstance(multiple_data_test_cases, dict):
            if (_list := len([data for data in multiple_data_test_cases.values() if isinstance(data, np.ndarray)]) > 0)\
                    and _list < len(multiple_data_test_cases):
                with pytest.raises(AssertionError):
                    plot_z_histogram_to_ax(ax1, multiple_data_test_cases)
                return

        plot_z_histogram_to_ax(ax1, multiple_data_test_cases, title="")
        auto_show(fig1)
        assert len(ax1.lines) == 0  # no Gauss fit (expected default behaviour)

        if len(multiple_data_test_cases) == 1:
            assert ax1.get_title().startswith("\u03BC=")  # default title for one data set shows mu=...
        else:
            assert ax1.get_title() == ""  # no title if no data or more than one dataset

        assert ax1.get_xlabel() == "z [\u03BCm]"  # default units should be µm; note:  µ == \u03BC is False!
        for container in ax1.containers:
            assert len(container.patches) == 200  # default n_bins should be 200

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

    def test_plot_z_histogram_to_ax__multiple(self, multiple_data_test_cases):
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, constrained_layout=True)
        pixel_width = None
        if isinstance(multiple_data_test_cases, dict):
            pixel_width = 0.5e-6
            # if (_list := len([data for data in multiple_data_test_cases.values() if isinstance(data, np.ndarray)]) > 0)\
            #         and _list < len(multiple_data_test_cases):
            #     with pytest.raises(AssertionError):
            #         plot_z_histogram_to_ax(ax1, multiple_data_test_cases)
            #     return

        plot_z_histogram_to_ax(ax1, multiple_data_test_cases, pixel_width=pixel_width, title="", add_norm=True)
        auto_show(fig1)
        assert isinstance(fig1, Figure)
        if isinstance(multiple_data_test_cases, np.ndarray) or isinstance(multiple_data_test_cases, GDEFSticher):
            assert len(fig1.axes[0].containers) == 1
            assert len(fig1.axes[0].lines) == 1
        else:
            assert len(fig1.axes[0].containers) == len(multiple_data_test_cases)
            assert len(fig1.axes[0].lines) != 2  # Gauss fits (add_norm=True)

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

    def test_create_z_histogram_plot__multiple(self, multiple_data_test_cases):
        labels = None
        pixel_width = 0.5e-6
        if isinstance(multiple_data_test_cases, list):
            labels = []
            for i, data in enumerate(multiple_data_test_cases):
                labels.append(f"{i} - {type(data).__name__}")

        fig1 = create_z_histogram_plot(multiple_data_test_cases, pixel_width=pixel_width, labels=labels, title="",
                                       add_norm=True)
        auto_show(fig1)
        assert len(fig1.axes[0].containers) == len(multiple_data_test_cases)
        assert len(fig1.axes[0].lines) == len(multiple_data_test_cases)  # Gauss fits (add_norm=True)


class Test1DPlotRMS:
    def test_plot_rms_to_ax(self):
        pass

    def test_create_rms_plot__default(self, data_test_cases):
        fig = create_rms_plot(data_test_cases)
        assert isinstance(fig, Figure)
        if isinstance(data_test_cases, np.ndarray):
            assert fig.axes[0].get_xlabel() == "x [px]"
        else:
            assert fig.axes[0].get_xlabel() == "x [\u03BCm]"
        assert fig.axes[0].legend_ is None
        auto_show(fig)

    def test_create_rms_plot__set_parameters(self, data_test_cases):
        pixel_width = 0.5e-9  # define a length scale for np.ndarray
        labels = f"{type(data_test_cases).__name__}"
        fig = create_rms_plot(data_test_cases, label_list=labels, pixel_width=pixel_width, moving_average_n=1,
                              subtract_average=True, units="nm")
        assert isinstance(fig, Figure)
        assert fig.axes[0].get_xlabel() == "x [nm]"
        assert fig.axes[0].legend_ is not None
        auto_show(fig)

    def test_create_rms_plot__multiple_default(self, multiple_data_test_cases):
        if isinstance(multiple_data_test_cases, dict):
            if (_list := len([data for data in multiple_data_test_cases.values() if isinstance(data, np.ndarray)]) > 0)\
                    and _list < len(multiple_data_test_cases):
                with pytest.raises(AssertionError):
                    create_rms_plot(multiple_data_test_cases)
                return

        fig = create_rms_plot(multiple_data_test_cases)
        assert len(multiple_data_test_cases) == len(fig.axes[0].lines)
        auto_show(fig)

    def test_create_rms_plot__multiple_set_parameter(self, multiple_data_test_cases):
        labels=None
        pixel_width = 0.5e-6
        if isinstance(multiple_data_test_cases, list):
            labels = [f"{type(data).__name__}" for data in multiple_data_test_cases]
        fig = create_rms_plot(multiple_data_test_cases, label_list=labels, pixel_width=pixel_width, moving_average_n=1,
                              subtract_average=False, units="nm")
        assert fig.axes[0].legend_ is not None or len(multiple_data_test_cases) == 0
        assert len(multiple_data_test_cases) == len(fig.axes[0].lines)
        assert fig.axes[0].get_xlabel() == "x [nm]"
        auto_show(fig)


class Test1DPlotRMSWithError:
    def test_create_rms_with_error_plot(self, data_test_cases):
        fig = create_rms_with_error_plot(data_test_cases)
        if isinstance(data_test_cases, np.ndarray):
            assert fig.axes[0].get_xlabel() == "x [px]"
        else:
            assert fig.axes[0].get_xlabel() == "x [\u03BCm]"
        auto_show(fig)

    def test_create_rms_with_error_plot__multiple(self, multiple_data_test_cases):
        if isinstance(multiple_data_test_cases, dict):
            if (_list := len([data for data in multiple_data_test_cases.values() if isinstance(data, np.ndarray)]) > 0)\
                    and _list < len(multiple_data_test_cases):
                with pytest.raises(AssertionError):
                    create_rms_with_error_plot(multiple_data_test_cases)
                return

        fig = create_rms_with_error_plot(multiple_data_test_cases)
        assert fig.axes[0].get_xlabel() == "x [\u03BCm]"
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
