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
from gdef_reporter.plotter_utils import plot_to_ax, create_plot, plot_z_histogram_to_ax

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
    yield rs.random((256, 256)) * 1e-6


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
def _test_case(request, gdef_measurement, random_ndarray2d_data, gdef_sticher):
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
    def test_plot_to_ax(self, _test_case):
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        plot_to_ax(ax1, _test_case, pixel_width=1.0)
        assert type(fig1) is Figure
        assert fig1.axes[1].get_title() == "nm"  # default unit for z-values should be nm
        auto_show(fig1)

        fig2, ax2 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        pixel_width = 5.0
        title = f"{type(_test_case).__name__}\npixel_width={pixel_width}"
        plot_to_ax(ax2, _test_case, pixel_width=pixel_width, title=title)
        assert ax2.get_title() == title
        auto_show(fig2)

        fig3, ax3 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE)
        z_factor = 1.0
        title = f"{type(_test_case).__name__}\nz_unit: [m] - z_factor={z_factor}"
        plot_to_ax(ax3, _test_case, pixel_width=5.0, z_unit="m", z_factor=z_factor, title=title)
        assert fig3.axes[1].get_title() == "m"
        auto_show(fig3)

    def test_create_plot(self, _test_case):
        fig1 = create_plot(_test_case, 1e-6,  "default value for cropped (True)", ORIGINAL_FIGURE_SIZE, ORIGINAL_DPI)
        assert np.any(comparison := (fig1.get_size_inches() < ORIGINAL_FIGURE_SIZE)) and not np.all(comparison)
        assert fig1.dpi == ORIGINAL_DPI
        auto_show(fig1)

        fig2 = create_plot(_test_case, 1e-6, "cropped=False", max_figure_size=ORIGINAL_FIGURE_SIZE, cropped=False)
        assert np.all(fig2.get_size_inches() == ORIGINAL_FIGURE_SIZE)
        auto_show(fig2)

    def test_plot_z_histogram_to_ax(self, _test_case):
        fig1, ax1 = plt.subplots(1, 1, dpi=ORIGINAL_DPI, figsize=ORIGINAL_FIGURE_SIZE, )
        plot_z_histogram_to_ax(ax1, _test_case, n_bins=20, units="nm")
        assert type(fig1) is Figure
        auto_show(fig1)



