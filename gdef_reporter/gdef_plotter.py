"""
GDEFPlotter is used to create matplotlib Figures for AFM measurements.
@author: Nathanael Jöhrmann
"""
from __future__ import annotations

import copy
from typing import Optional, List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from gdef_reader.utils import create_absolute_gradient_array
from gdef_reporter.plotter_styles import PlotterStyle, get_plotter_style_rms, get_plotter_style_sigma
from gdef_reporter.plotter_utils import create_plot, create_rms_plot, create_summary_plot, \
    create_rms_with_error_plot, _extract_ndarray_and_pixel_width

if TYPE_CHECKING:
    from gdef_reporter.plotter_utils import DataObject, DataObjectList


class GDEFPlotter:
    def __init__(self, figure_size=(12, 6), dpi: int = 300, auto_show: bool = False):
        """

        :param figure_size: figure size for created Fiugres
        :param dpi: dpi for created Figures
        :param auto_show: automatically call figure.show(), when a figure is created. If None, class attribute is used.
        """
        self._dpi = dpi
        self._figure_size = figure_size
        self.plotter_style_rms: PlotterStyle = get_plotter_style_rms(dpi=dpi, figure_size=figure_size)
        self.plotter_style_sigma: PlotterStyle = get_plotter_style_sigma(dpi=dpi, figure_size=figure_size)
        self.auto_show = auto_show

        # todo: implement auto save functionality
        self.auto_save = False
        self.save_path = None
        # method to save figures as pdf or png

    @property
    def dpi(self):
        return self._dpi

    @property
    def figure_size(self):
        return self._figure_size

    @dpi.setter
    def dpi(self, value: int):
        self.set_dpi_and_figure_size(dpi=value)

    @figure_size.setter
    def figure_size(self, value: tuple[float, float]):
        self.set_dpi_and_figure_size(figure_size=value)

    def set_dpi_and_figure_size(self, dpi: Optional[int] = None, figure_size: Optional[tuple[float, float]] = None):
        """
        Used to set dpi and (max.) figure size for created matpoltlib Figures. This includes updating the PlotterStyles.
        :param dpi:
        :param figure_size:
        :return: None
        """
        if dpi is None:
            dpi = self.dpi
        if figure_size is None:
            figure_size = self.figure_size
        self._dpi = dpi
        self._figure_size = figure_size
        self.plotter_style_rms.set(dpi=dpi, figure_size=figure_size)
        self.plotter_style_sigma.set(dpi=dpi, figure_size=figure_size)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ 2D area plots ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_plot(self, data_object: DataObject,
                    pixel_width: Optional[float] = None,
                    title: str = "",
                    cropped=True) -> Optional[Figure]:
        """
        Create a matplotlib Figure using the 2D array values as input data.
        :param data_object: data object containing a np.ndarray (2D)
        :param pixel_width: [m]
        :param title: optional Figure title (not Axes subtitle)
        :param cropped: Crop the result Figure (default is True). Useful if aspect ratio of Figure and plot differ.
        :return: Figure
        """
        result = create_plot(data_object, pixel_width, title, self.figure_size, self.dpi, cropped=cropped)

        self._auto_show_figure(result)
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ 1D plots over x -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_rms_per_column_plot(self, data_object_list: DataObjectList, pixel_width: Optional[float] = None,
                                   title: str = "", moving_average_n: int = 1, x_offset=0, subtract_average=True) \
            -> Figure:
        """
        Calculate root mean square roughness for each column in values and plot them over x. If moving_average_n
        is larger than 1, the RMS values are averaged over n columns.
        :param data_object_list:
        :param pixel_width: in meter
        :param title: optional figure title
        :param moving_average_n: number of columns for moving average
        :param x_offset:
        :param subtract_average:
        :return: matplotlib Figure
        """
        result = create_rms_plot(data_object_list, pixel_width, title=title, moving_average_n=moving_average_n,
                                 x_offset=x_offset, subtract_average=subtract_average,
                                 plotter_style=self.plotter_style_rms)
        self._auto_show_figure(result)
        return result

    def _create_absolute_gradient_rms_plot(self, data_object: DataObject, cutoff_percent_list, pixel_width=None,
                                           title=None, moving_average_n=1, x_offset=0, subtract_average=False)\
            -> Figure:
        """
        Creates a plot with a curve for each value in cutoff_percent_list, showing rms(abs(grad(z))) as
        moving average over moving_average_n columns.
        Candidate to become deprecated!
        :param data_object:
        :param cutoff_percent_list:
        :param pixel_width:
        :param title:
        :param moving_average_n:
        :return: Figure
        """
        grad_style = copy.deepcopy(self.plotter_style_rms)
        ndarray2d_data, pixel_width = _extract_ndarray_and_pixel_width(data_object, pixel_width)

        grad_data_list = []
        label_list = []
        for i, percent in enumerate(cutoff_percent_list):
            grad_data_list.append(create_absolute_gradient_array(ndarray2d_data, percent / 100.0))
            label_list.append(f"{percent}%")

        result = create_rms_plot(grad_data_list, pixel_width, label_list, title=title,
                                 moving_average_n=moving_average_n,
                                 x_offset=x_offset, subtract_average=subtract_average,
                                 plotter_style=grad_style)

        grad_style.set(y_label=f"rms(abs(grad(z)))\n(moving avg. n = {moving_average_n} column(s))", y_unit="")
        grad_style.set_format_to_ax(result.axes[0])
        result.tight_layout()
        self._auto_show_figure(result)
        return result

    def _create_absolute_gradient_maps_plot(self, values: np.ndarray, cutoff_percent_list: List[int],
                                            title=None, nan_color='red') -> Figure:
        """
        Creates a matplotlib figure, to show the influence of different cutoff values. The omitted values are represented
        in the color nan_color (default is red).
        :param values:
        :param cutoff_percent_list:
        :param title:
        :param nan_color:
        :return:
        """
        result, ax_list_cutoff = plt.subplots(len(cutoff_percent_list), 1,
                                              figsize=(self.figure_size[0], len(cutoff_percent_list)))

        cmap_gray_red_nan = copy.copy(plt.cm.gray)  # use copy to prevent unwanted changes to other plots somewhere else
        cmap_gray_red_nan.set_bad(color=nan_color)

        for i, percent in enumerate(cutoff_percent_list):
            absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
            ax_list_cutoff[i].imshow(absolut_gradient_array, cmap_gray_red_nan)
            ax_list_cutoff[i].set_title(f'gradient cutoff {percent}%')
            ax_list_cutoff[i].set_axis_off()
        if title:
            result.suptitle(title)
        result.tight_layout()
        self._auto_show_figure(result)
        return result

    def create_summary_plot(self, data_object_list: DataObjectList, pixel_width=None):  # , figure_size=(16, 10)):
        """
        Creates a Figure with maps for each DataObject in data_object_list.
        :param data_object_list:
        :return:
        """
        result = create_summary_plot(data_object_list, pixel_width=pixel_width,
                                     figure_size=self.figure_size, dpi=self.dpi)
        self._auto_show_figure(result)
        return result

    def create_rms_with_error_plot_from_sticher_dict(self, sticher_dict, average_n=8):
        result = create_rms_with_error_plot(sticher_dict, average_n)
        if self.auto_show:
            result.show()
        return result

    def _auto_show_figure(self, fig):
        if self.auto_show:
            fig.show()
