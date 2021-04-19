"""
@author: Nathanael Jöhrmann
"""
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import signal

from gdef_reader.gdef_measurement import GDEFMeasurement


class GDEFSticher:
    """
    GDEFSticher combines/stiches several AFM area-measurements using cross-corelation to find the best fit.
    To reduce calculation time, the best overlap position is only searched in a fraction of the measurement area
    (defined by parameter initial_x_offset_fraction), and each measutrement is added to the right side.
    Make sure the given list of measurements is ordered from left to right, otherwise wrong results are to be expected.
    To evaluate the stiching, show_control_figures can be set to True. This creates a summary image
    for each stiching step (using matplotlib plt.show()).

    :InstanceAttributes:
    measurements: list of GDEFMeasurements used for stiching
    values: np.ndarray with stiched data
    pixel_width: Pixel width taken from first GDEFMeasurement in measurements (varying pixel sizes are not supported).
    :EndInstanceAttributes:
    """

    def __init__(self, measurements: List[GDEFMeasurement],
                 initial_x_offset_fraction: float = 0.35, show_control_figures: bool = False):
        """
        :param measurements:
        :param initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
        :param show_control_figures:
        """
        self.measurements = measurements
        self.values = None
        self.pixel_width = self.measurements[0].settings.pixel_width
        for measurement in self.measurements:
            if measurement.settings.pixel_width != self.pixel_width:
                warnings.warn(f"Measurement {measurement.name} has a different pixel_width than used for GDEFSticher!")

        self.stich(initial_x_offset_fraction, show_control_figures)

    def stich(self, initial_x_offset_fraction: float = 0.35, show_control_figures: bool = False) -> np.ndarray:
        """
        Stiches a list of GDEFMeasurement.values using cross-correlation.
        :param initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
        :return: stiched np.ndarray
        """
        result = self.measurements[0].values
        x_offset_right = round(result.shape[1] * initial_x_offset_fraction)
        for measurement in self.measurements[1:]:
            result = self._stich(result, measurement.values, result.shape[1] - x_offset_right, show_control_figures)
        self.values = result
        return result

    def _stich(self, data01: np.ndarray, data02:np.ndarray, data01_x_offset: int, show_control_figures: bool = True) -> np.ndarray:
        """
        Stiches two GDEFMeasurement.values using cross-correlation.
        :param data01:
        :param data02:
        :param data01_x_offset: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
        :return: np.ndarray (2D)
        """
        # data01[:] = 10
        # data02[:] = 10
        data02_x_offset_right = data01.shape[1] - data01_x_offset
        correlation = signal.correlate2d(data01[:, data01_x_offset:],
                                         data02[:, :data02_x_offset_right])  #, boundary="wrap")  # using "wrap" ensures, that x, y below can be used directly

        reduced_correlation = correlation[:, data02_x_offset_right:]  # make sure, data02 is appended on right side
                                                                      # this reduces risk of wrong stiching, but measurements have to be in right order
        # max_overlap_area = (reduced_correlation.shape[1]+1) * (reduced_correlation.shape[0]+1)
        # # overlap_area =
        # with np.nditer(reduced_correlation, op_flags=['readwrite'], flags=['multi_index']) as it:
        #     for x in it:
        #         #print("%d <%s>" % (x, it.multi_index), end=' ')
        #         iy, ix = it.multi_index
        #         cy, cx = reduced_correlation.shape
        #         # x[...] = x * max_overlap_area / ((cx-ix) * min(iy+1, cy-iy))  # norm by oberlapping area
        #         # x[...] = x * (reduced_correlation.shape[0]+1) / min(iy+1, cy-iy)  # norm only with y
        #         x[...] = x * (reduced_correlation.shape[1]+1) / (cx-ix)  # norm only with x
        #         #print("%d <%s>" % (x, it.multi_index), end=' ')

        y, x = np.unravel_index(np.nanargmax(reduced_correlation), reduced_correlation.shape)  # find (first) best match
        y, x = y - data02.shape[0] + 1, x + 1 + data01_x_offset  # - data02_x_offset_right)  # test with two identical datasets -> should give: y, x = 0, 0
        # ax_orig.plot(x, y, 'ro')

        data01_x0 = - min(0, x)
        data01_y0 = - min(0, y)
        data02_x0 = max(0, x)
        data02_y0 = max(0, y)

        data01_height = data01.shape[0] + data01_y0
        data01_width = data01.shape[1] + data01_x0

        data02_height = data02.shape[0] + data02_y0
        data02_width = data02.shape[1] + data02_x0

        result = np.full([max(data01_height, data02_height), max(data01_width, data02_width)], np.nan)

        result[data01_y0:data01_height, data01_x0:data01_width] = data01
        result[data02_y0:data02_height, data02_x0:data02_width] = data02

        if show_control_figures:
            plt = self._create_stich_control_figure(data01, data02, reduced_correlation, result)
            plt.show()
        return result

    # def create_cropped_figure(self, max_figure_size: Tuple[float, float] = (20, 10), dpi: int = 300) -> Figure:
    #     # todo: something is broken here
    #     create_cropped_plot(self.stiched_data, self.pixel_width, max_figure_size)

    def _create_stich_control_figure(self, data01: np.ndarray, data02: np.ndarray,
                                     correlation: np.ndarray, stiched_data: np.ndarray) -> Figure:
        result, (ax_orig, ax_template, ax_corr, ax_stich) = plt.subplots(4, 1, figsize=(6, 20))

        ax_orig.imshow(data01, cmap='gray')
        ax_orig.set_title('data01')
        ax_orig.set_axis_off()

        ax_template.imshow(data02, cmap='gray')
        ax_template.set_title('data02')
        ax_template.set_axis_off()

        ax_corr.imshow(correlation, cmap='gray')
        ax_corr.set_title('Cross-correlation')
        ax_corr.set_axis_off()

        ax_stich.set_title('stiched')
        ax_stich.set_axis_off()
        ax_stich.imshow(stiched_data, cmap='gray')

        return result

