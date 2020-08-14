import copy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
# from scipy import misc
import numpy as np
from scipy import signal

from gdef_reader.gdef_measurement import GDEFMeasurement
from gdef_reader.utils import load_pygdf_measurements
# import png

class GDEFSticher:
    def __init__(self, measurements: List[GDEFMeasurement], initial_x_offset_fraction = 0.35, show_control_figures=False):
        self.measurements = measurements
        self.stiched_data = None

        self.stich(initial_x_offset_fraction, show_control_figures)

    def stich(self, initial_x_offset_fraction: float = 0.35, show_control_plots: bool = True) -> np.ndarray:
        """
        Stiches a list of GDEFMeasurement.values using cross-correlation.
        :param values_list: List of GDEFMeasurement.values
        :param initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
        :return: stiched np.ndarray
        """
        result = self.measurements[0].values
        x_offset_right = round(result.shape[1] * initial_x_offset_fraction)
        for measurement in self.measurements[1:]:
            result = self._stich(result, measurement.values, result.shape[1] - x_offset_right, show_control_plots)
        return result

    def _stich(self, data01, data02, data01_x_offset, show_control_figures = True):
        """
        Stiches two GDEFMeasurement.values using cross-correlation.
        :param data01:
        :param data02:
        :param data01_x_offset: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
        :return: np.2darray
        """
        data02_x_offset_right = data01.shape[1] - data01_x_offset
        correlation = signal.correlate2d(data01[:, data01_x_offset:],
                                         data02[:, :data02_x_offset_right])  # , boundary="wrap")  # using "wrap" ensures, that x, y below can be used directly

        reduced_correlation = correlation[:, data02_x_offset_right:]  # make sure, data02 is appended on right side
                                                                      # this reduces risk of wrong stiching, but measurements have to be in right order

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
            self.createstich_control_figure(data01, data02, correlation)
        return result

    def createstich_control_figure(self, data01: np.ndarray, data02: np.ndarray, correlation: np.ndarray):
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
        ax_stich.imshow(result, cmap='gray')

        return result
