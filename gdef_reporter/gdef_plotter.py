"""
@author: Nathanael Jöhrmann
"""
import copy
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# from afm_tools.gdef_sticher import GDEFSticher
from afm_tools.gdef_sticher import GDEFSticher
from gdef_reader.utils import create_xy_rms_data, create_absolute_gradient_array, get_mu_sigma_moving_average, \
    get_mu_sigma
from gdef_reporter._code_utils import ClassOrInstanceMethod
from gdef_reporter.plotter_styles import PlotterStyle, get_plotter_style_rms, get_plotter_style_sigma
from gdef_reporter.plotter_utils import create_plot_from_sticher, plot_sticher_to_ax, plot_from_ndarray_to_ax, \
    create_plot_from_ndarray


class GDEFPlotter:
    auto_show = False  # used for classmethod and as default for instance auto_show

    def __init__(self, figure_size=(12, 6), dpi=300, auto_show=None):
        """

        :param figure_size: figure size for created Fiugres
        :param dpi: dpi for created Figures
        :param auto_show: automatically call figure.show(), when a figure is created. If None, class attribute is used.
        """
        self._dpi = dpi
        self._figure_size = figure_size
        self.plotter_style_rms: PlotterStyle = get_plotter_style_rms(dpi=dpi, figure_size=figure_size)
        self.plotter_style_sigma: PlotterStyle = get_plotter_style_sigma(dpi=dpi, figure_size=figure_size)
        if auto_show is None:
            self.auto_show = GDEFPlotter.auto_show
        else:
            self.auto_show = auto_show

    @property
    def dpi(self):
        return self._dpi

    @property
    def figure_size(self):
        return self._figure_size

    @dpi.setter
    def dpi(self, value):
        self.set_dpi_and_figure_size(dpi=value)

    @figure_size.setter
    def figure_size(self, value):
        self.set_dpi_and_figure_size(figure_size=value)

    def set_dpi_and_figure_size(self, dpi=None, figure_size=None):
        if dpi is None:
            dpi = self.dpi
        if figure_size is None:
            figure_size = self.figure_size
        self._dpi = dpi
        self._figure_size = figure_size
        self.plotter_style_rms: PlotterStyle = get_plotter_style_rms(dpi=dpi, figure_size=figure_size)
        self.plotter_style_sigma: PlotterStyle = get_plotter_style_sigma(dpi=dpi, figure_size=figure_size)

    def create_surface_figure(self, values: np.ndarray, pixel_width, title=None, cropped=True) -> Optional[Figure]:
        if values is None:
            return
        fig = create_plot_from_ndarray(values, pixel_width, title, cropped, self.figure_size, self.dpi)

        self._auto_show_figure(fig)
        return fig

    def create_rms_per_column_figure(self, values: np.ndarray, pixel_width, title=None, moving_average_n=1) -> Figure:
        """
        :param values: 2D array
        :param pixel_width: in meter
        :param title: optional figure title
        :param moving_average_n: number of columns for moving average
        :return: matplotlib Figure
        """
        x_pos, y_rms = create_xy_rms_data(values, pixel_width, moving_average_n)
        result, ax_rms = self.plotter_style_rms.create_preformated_figure()

        ax_rms.plot(x_pos, y_rms, **self.plotter_style_rms.graph_styler.dict)  # 'r')
        if title:
            info = f"moving average n={moving_average_n} ({moving_average_n * pixel_width * 1e6:.1f} µm)"
            result.suptitle(f'{title}\n{info}')  # , fontsize=16)
        result.tight_layout()
        self._auto_show_figure(result)
        return result

    def create_absolute_gradient_rms_figure(self, values: np.ndarray, cutoff_percent_list, pixel_width, title=None,
                                            moving_average_n=1) -> Figure:
        """
        Creates a plot with a curve for each value in cutoff_percent_list, showing rms(abs(grad(z))) as
        moving average over moving_average_n columns.
        Candidate to become deprecated!
        :param values:
        :param cutoff_percent_list:
        :param pixel_width:
        :param title:
        :param moving_average_n:
        :return: Figure
        """
        result, (ax_gradient_rms) = plt.subplots(1, 1, figsize=self.figure_size)
        ax_gradient_rms.set_xlabel("[µm]")
        ax_gradient_rms.set_ylabel(f"rms(abs(grad(z))) (moving avg. n = {moving_average_n} column(s))")

        for i, percent in enumerate(cutoff_percent_list):
            absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
            x_pos, y_gradient_rms = create_xy_rms_data(absolut_gradient_array, pixel_width, moving_average_n)
            ax_gradient_rms.plot(x_pos, y_gradient_rms, label=f"{percent}%")
        ax_gradient_rms.legend()
        if title:
            result.suptitle(title)
        result.tight_layout()
        self._auto_show_figure(result)
        return result

    @ClassOrInstanceMethod
    def _auto_show_figure(cls, instance, fig):
        """Parameters cls and instance are set via decorator ClassOrInstance. So first parameter when calling is fig."""
        auto_show = instance.auto_show if instance else cls.auto_show
        if auto_show:
            fig.show()

    def create_sigma_moving_average_figure(self, sticher_dict: Dict[str, GDEFSticher], moving_average_n=200, step=1):
        """

        :param sticher_dict:
        :param moving_average_n:
        :param step: col step value between moving average values (default 1; moving avg. is calculated for each col)
        :return:
        """
        # todo: how to make this work with [GDEFMeasurment] too? Split sticher_dict in List[ndarray] and list[label]?
        x_pos = []
        y_sigma = []
        pixel_width_in_um = None

        graph_styler = self.plotter_style_sigma.graph_styler.reset()
        result, ax_sigma = self.plotter_style_sigma.create_preformated_figure()

        for key, sticher in sticher_dict.items():
            x_pos = []
            y_sigma = []
            pixel_width_in_um = sticher.pixel_width * 1e6
            for i in range(0, sticher.stiched_data.shape[1] - moving_average_n, step):
                x_pos.append((i + max(moving_average_n - 1, 0) / 2.0) * pixel_width_in_um)

            _, y_sigma = get_mu_sigma_moving_average(sticher.stiched_data * 1e6, moving_average_n, step)

            ax_sigma.plot(x_pos, y_sigma, **graph_styler.dict, label=key)
            graph_styler.next_style()

        ax_sigma.legend()
        moving_window_in_um = pixel_width_in_um * moving_average_n
        result.suptitle(f"Moving average n={moving_average_n} ({moving_window_in_um:.1f} µm)")

        result.tight_layout()
        self._auto_show_figure(result)
        return result

    def create_absolute_gradient_figures(self, values: np.ndarray, cutoff_percent_list,
                                         title=None, nan_color='red') -> Figure:
        """
        Creates a matplotlib figure, to show the influence of different cutoff values. The omitted values are represented
        in the color nan_color (default is red).
        :param values:
        :param cutoff_percent_list:
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

    def create_stich_summary_figure(self, sticher_dict):  # , figure_size=(16, 10)):
        """
        Creates a Figure with stiched maps for each GDEFSticher in sticher_dict. The keys in sticher_dict
        are used as titles for the corresponding Axes.
        :param sticher_dict:
        :return:
        """
        n = len(sticher_dict)
        if n == 0:
            return plt.subplots(1, figsize=self.figure_size, dpi=300)

        # todo refactor in a seperate function
        optimal_ratio = self.figure_size[0] / self.figure_size[1]
        dummy_fig = create_plot_from_sticher(list(sticher_dict.values())[0], title='dummy',
                                             max_figure_size=self.figure_size)  # measurements[0].create_plot()
        single_plot_ratio = dummy_fig.get_figwidth() / dummy_fig.get_figheight()
        optimal_ratio /= single_plot_ratio

        possible_ratios = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j >= n:
                    x, y = i, j
                    possible_ratios.append((x, y))
                    break

        # sort ratios by best fit to optimal ratio:
        possible_ratios[:] = sorted(possible_ratios, key=lambda ratio: abs(ratio[0] / ratio[1] - optimal_ratio))
        best_ratio = possible_ratios[0][1], possible_ratios[0][0]

        result, ax_list = plt.subplots(*best_ratio, figsize=self.figure_size, dpi=300)
        for i, key in enumerate(sticher_dict):
            y = i // best_ratio[0]
            x = i - (y * best_ratio[0])
            if (best_ratio[1] > 1) and (best_ratio[0] > 1):
                plot_sticher_to_ax(sticher_dict[key], ax_list[x, y], title=key)
            elif best_ratio[0] > 1:
                plot_sticher_to_ax(sticher_dict[key], ax_list[x], title=key)
            elif best_ratio[1] > 1:
                plot_sticher_to_ax(sticher_dict[key], ax_list[y], title=key)
            else:
                plot_sticher_to_ax(sticher_dict[key], ax_list, title=key)
        i = len(sticher_dict)
        while i < best_ratio[0] * best_ratio[1]:
            y = i // best_ratio[0]
            x = i - (y * best_ratio[0])
            ax_list[x, y].set_axis_off()
            i += 1
        result.tight_layout()
        self._auto_show_figure(result)
        return result

    def create_rms_with_error_fig(self, sticher_dict, average_n=8 * 160, step=1):
        graph_styler = self.plotter_style_sigma.graph_styler.reset()
        result, ax_rms = self.plotter_style_sigma.create_preformated_figure()

        for key, sticher in sticher_dict.items():
            z_data = sticher.stiched_data

            # get mu for every column first:
            sigma_col_list = []
            for i in range(0, z_data.shape[1]):
                _, sigma_col = get_mu_sigma(z_data[:, i:i + 1])
                sigma_col_list.append(sigma_col)

            x_pos = []
            y_rms = []
            y_error = []
            pixel_width_in_um = sticher.pixel_width * 1e6
            for i in range(0, z_data.shape[1] - average_n, average_n):  # step):
                x_pos.append((i + max(average_n - 1, 0) / 2.0) * pixel_width_in_um)

                mu_rms, sigma_rms = get_mu_sigma(np.array(sigma_col_list[i:i + average_n]))
                y_rms.append(mu_rms * 1e6)
                y_error.append(sigma_rms * 1e6)
            style_dict = {
                "fmt": 'o',
                "elinewidth": 0.6,
                "capsize": 2.0,
                "markersize": 5,
                "color": graph_styler.dict["color"]
            }
            ax_rms.errorbar(x_pos, y_rms, yerr=y_error, label=key,
                            **style_dict)  # **graph_styler.dict, label=key)  #fmt='-o')  # **graph_styler.dict
            graph_styler.next_style()
        # ax_rms.set_title(f"window width = {moving_average_n*pixel_width_in_um:.1f}")

        name = list(sticher_dict.keys())[0]
        name.replace(",", "").replace(" ", "_")
        result.tight_layout()

        ax_rms.legend()
        # legend_handles, legend_labels = ax_rms.get_legend_handles_labels()
        # order = [2, 0, 1]
        # ax_rms.legend([legend_handles[idx] for idx in order], [legend_labels[idx] for idx in order], fontsize=8)
        if self.auto_show:
            result.show()
        return result

    def create_plot_from_sticher(self, sticher: GDEFSticher, title=''):
        """Calls create_plot_from_sticher() from plotter_utils using self.figure_size and self.dpi."""
        fig = create_plot_from_sticher(sticher, title, max_figure_size=self.figure_size, dpi=self.dpi)
        self._auto_show_figure(fig)
        return fig

    # @classmethod
    # def plot_sticher_to_axes(cls, sticher: GDEFSticher, ax: Axes, title=''):
    #     """
    #     Deprecated - please use plot_surface_to_axes() from plotter_utils.
    #     """
    #     print("GDEFPlotter.set_topography_to_axes is deprecated. Please use plot_surface_to_axes() from plotter_utils.")
    #     plot_surface_to_axes(ax=ax, values=sticher.stiched_data, pixel_width=sticher.pixel_width, title=title)

    # @classmethod
    # def plot_surface_to_axes(cls, ax: Axes, values: np.ndarray, pixel_width: float,
    #                          title="", z_unit="nm", z_factor=1e9):
    #     """
    #     Deprecated - please use plot_surface_to_axes() from plotter_utils.
    #     """
    #     print("GDEFPlotter.plot_surface_to_axes is deprecated. Please use plot_surface_to_axes() from plotter_utils.")
    #     plot_surface_to_axes(ax=ax, values=values, pixel_width=pixel_width,
    #                          title=title, z_unit=z_unit, z_factor=z_factor)
