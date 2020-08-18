from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gdef_reader.utils import create_xy_rms_data, create_absolute_gradient_array


class GDEFPlotter:
    def __init__(self, figure_size=(12, 6), dpi=300, auto_show=False):
        """

        :param figure_size:
        :param dpi:
        :param auto_show: automatically call figure.show(), when a figure is created
        """
        self.dpi = dpi
        self.figure_size = figure_size
        self.auto_show = auto_show

    def create_surface_figure(self, values: np.ndarray, pixel_width, cropped=True) -> Optional[Figure]:
        if values is None:
            return

        figure_max, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        self.plot_surface_to_axes(ax, values, pixel_width)
        figure_max.tight_layout()
        if not cropped:
            self._auto_show_figure(figure_max)
            return figure_max

        tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
        figure_tight, ax = plt.subplots(figsize=tight_bbox.size, dpi=self.dpi)
        self.plot_surface_to_axes(ax, values, pixel_width)
        self._auto_show_figure(figure_tight)
        return figure_tight

    def create_rms_per_column_figure(self, values: np.ndarray, pixel_width, title=None, moving_average_n=1) -> Figure:
        """
        :param values: 2D array
        :param pixel_width: in meter
        :param title: optional figure title
        :param moving_average_n: number of columns for moving average
        :return: matplotlib Figure
        """
        x_pos, y_rms = create_xy_rms_data(values, pixel_width, moving_average_n)
        result, (ax_rms) = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        ax_rms.plot(x_pos, y_rms, 'r')
        ax_rms.set_xlabel("[µm]")
        ax_rms.set_ylabel(f"root mean square (moving average over {moving_average_n} column(s))")
        if title:
            result.suptitle(f'{title}', fontsize=16)
        self._auto_show_figure(result)
        return result

    def create_absolute_gradient_rms_figure(self, values: np.ndarray, cutoff_percent_list, pixel_width, moving_average_n=1) -> Figure:
        result, (ax_gradient_rms) = plt.subplots(1, 1, figsize=self.figure_size)
        ax_gradient_rms.set_xlabel("[µm]")
        ax_gradient_rms.set_ylabel(f"rms(abs(grad(surface)))) (moving average over {moving_average_n} column(s))")

        for i, percent in enumerate(cutoff_percent_list):
            absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
            x_pos, y_gradient_rms = create_xy_rms_data(absolut_gradient_array, pixel_width, moving_average_n)
            ax_gradient_rms.plot(x_pos, y_gradient_rms, label=f"{percent}%")
        ax_gradient_rms.legend()
        self._auto_show_figure(result)
        return result

    def _auto_show_figure(self, fig):
        if self.auto_show:
            fig.show()

    @classmethod
    def plot_surface_to_axes(cls, ax: Axes, values: np.ndarray, pixel_width: float):
        """
        Plot surface-values to given ax. Necessary, to use figures with subplots effectivly.
        """
        def extent_for_plot(shape, pixel_width):
            width_in_um = shape[1] * pixel_width * 1e6
            height_in_um = shape[0] * pixel_width * 1e6
            return [0, width_in_um, 0, height_in_um]

        extent = extent_for_plot(values.shape, pixel_width)
        im = ax.imshow(values * 1e9, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
        # ax.set_title(self.comment)  # , pad=16)
        ax.set_xlabel("µm", labelpad=1.0)
        ax.set_ylabel("µm", labelpad=1.0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_title("nm", y=1)  # bar.set_label("nm")
        plt.colorbar(im, cax=cax)

