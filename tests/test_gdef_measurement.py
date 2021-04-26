"""
This file contains tests for gdef_measurement.py.
@author: Nathanael Jöhrmann
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from afm_tools.background_correction import BGCorrectionType


def auto_show_fig(fig):
    flag = False  # set True to control created Figures visually
    if flag:
        fig.show()


@pytest.fixture(scope='session')
def gdef_settings(gdef_measurement):
    gdef_settings = gdef_measurement.settings
    yield gdef_settings


class TestGDEFSettings:
    def test_pixel_width(self, gdef_settings):
        assert np.isclose(gdef_settings.pixel_width, 1.9531249506599124e-07)

    def test_pixel_height(self, gdef_settings):
        assert np.isclose(gdef_settings.pixel_height, 1.9531249506599124e-07)

    def test_pixel_area(self, gdef_settings):
        assert np.isclose(gdef_settings.pixel_area(), 3.8146970728902854e-14)
        assert np.isclose(gdef_settings.pixel_area(), gdef_settings.pixel_width * gdef_settings.pixel_height)

    def test_size_in_um_for_plot(self, gdef_settings):
        assert gdef_settings.size_in_um_for_plot() == (0.0, 49.99999873689376, 0.0, 12.49999968422344)

    def test_shape(self, gdef_settings):
        assert gdef_settings.shape() == (256, 64)

    def test__data_type_info(self, gdef_settings):
        assert gdef_settings._data_type_info() == ('topography', 'nm', 1e9)


class TestGDEFMeasurement:
    def test_name_property(self, gdef_measurement):
        assert gdef_measurement.name.startswith("example_01_block_")
        with pytest.raises(AttributeError):
            gdef_measurement.name = "name should be a read-only-property"

    def test_values_original_property(self, gdef_measurement):
        original_data = gdef_measurement.values_original.copy()

        with pytest.raises(AttributeError):
            gdef_measurement.values_original = "values_original should be a read-only-property"
        with pytest.raises(ValueError):
            gdef_measurement.values_original[0, 0] = 5  # should be read-only

        gdef_measurement.values[0, 0] = 5  # change of values should not change original_values
        assert np.all(gdef_measurement.values_original == original_data)

    def test_save_as_pickle(self, gdef_measurement):
        pass

    def test_load_from_pickle(self, gdef_measurement):
        pass

    def test_save_png(self, gdef_measurement):
        pass

    def test_set_topography_to_axes(self, gdef_measurement):
        fig, axes = plt.subplots(2, 1, figsize=(4, 3.5), dpi=96)
        gdef_measurement.set_topography_to_axes(axes[0])
        assert fig.axes[0].get_title() == "topography"
        gdef_measurement.set_topography_to_axes(axes[1], add_id=True)
        assert fig.axes[1].get_title().endswith(": topography")  # ID depends on how many data was imported -> endswith
        auto_show_fig(fig)

    def test_create_plot(self, gdef_measurement):
        max_figure_size = (4, 2.5)
        dpi = 123
        original_comment = gdef_measurement.comment

        # default figure with empty comment
        gdef_measurement.comment = ""
        fig = gdef_measurement.create_plot()
        auto_show_fig(fig)
        assert fig.axes[0].get_title() == "topography"

        gdef_measurement.comment = "title with ID"
        fig = gdef_measurement.create_plot(max_figure_size=max_figure_size, dpi=dpi, add_id=True)
        auto_show_fig(fig)
        assert fig.axes[0].get_title() == f"{gdef_measurement.gdf_block_id}: {gdef_measurement.comment}"
        assert fig.dpi == dpi
        assert np.all(fig.get_size_inches() <= max_figure_size)

        # don't trim figure
        fig = gdef_measurement.create_plot(max_figure_size=max_figure_size, dpi=dpi, add_id=True, trim=False)
        auto_show_fig(fig)
        assert np.all(fig.get_size_inches() == max_figure_size)

        gdef_measurement.comment = "title without ID"
        fig = gdef_measurement.create_plot(max_figure_size=max_figure_size, dpi=dpi, add_id=False)
        auto_show_fig(fig)
        assert fig.axes[0].get_title() == gdef_measurement.comment

        gdef_measurement.comment = original_comment

    def test_correct_background(self, gdef_measurement):
        gdef_measurement.correct_background(BGCorrectionType.raw_data)
        assert np.all(gdef_measurement.values == gdef_measurement.values_original)
        assert gdef_measurement.background_correction_type == BGCorrectionType.raw_data

        gdef_measurement.correct_background(BGCorrectionType.legendre_1)
        assert np.all(gdef_measurement.values != gdef_measurement.values_original)

    def test_get_summary_table_data(self, gdef_measurement):
        table_data = [
            ['source channel', 11],
            ['retrace', 0],
            ['missing lines', 0],
            ['max width [m]', '5.00e-05'],
            ['max height [m]', '1.25e-05'],
            ['scan speed [µm/s]', '50'],
            ['name', 'example_01_block_002']  # can't test block_id - ID depends on gdef_import history
        ]

        assert gdef_measurement.get_summary_table_data()[:6] == table_data[:6]
