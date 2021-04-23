"""
@author: Nathanael Jöhrmann
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


class PlotterStyle:
    def __init__(self, dpi: Optional[int] = None, figure_size: Optional[Tuple[float, float]] = None):
        self.dpi = dpi
        self.figure_size = figure_size

        self._x_label = None
        self._y_label = None
        self._x_unit = None
        self._y_unit = None
        self.ax_title = None
        self.grid = None

        # self.ax_styler = AxStyler()
        self.graph_styler = None

    @property
    def x_label(self):
        if self._x_label is None and self._x_unit is None:
            return None
        result = ""
        if self._x_label:
            result += self._x_label
        if self._x_unit is not None:
            result += f" [{self._x_unit}]"
        return result

    @property
    def x_label(self):
        if self._x_label is None and self._x_unit is None:
            return None
        result = ""
        if self._x_label:
            result += self._x_label
        if self._x_unit is not None:
            result += f" [{self._x_unit}]"
        return result

    @property
    def y_label(self):
        if self._y_label is None and self._y_unit is None:
            return None
        result = ""
        if self._y_label:
            result += self._y_label
        if self._y_unit is not None:
            result += f" [{self._y_unit}]"
        return result

    def set(self, dpi=None, figure_size=None, x_label=None, y_label=None,
            x_unit = None, y_unit = None,
            ax_title=None, grid=None) -> None:
        """Set values for ..."""

        if dpi is not None:
            self.dpi = dpi
        if figure_size is not None:
            self.figure_size = figure_size

        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label

        if x_unit is not None:
            self._x_unit = x_unit
        if y_unit is not None:
            self._y_unit = y_unit

        if ax_title is not None:
            self.ax_title = ax_title
        if grid is not None:
            self.grid = grid

    def set_format_to_ax(self, ax: Axes):
        if self._x_label is not None:
            ax.set_xlabel(self.x_label)
        if self._y_label is not None:
            ax.set_ylabel(self.y_label)
        if self.ax_title is not None:
            ax.set_title(self.ax_title)
        if self.grid is not None:
            ax.grid(self.grid)

    def create_preformated_figure(self, nrows=1, ncols=1):
        fig, axs = plt.subplots(nrows, ncols, figsize=self.figure_size, dpi=self.dpi)

        if (nrows == 1) and (ncols == 1):
            self.set_format_to_ax(axs)
        elif (nrows == 1) or (ncols == 1):
            for ax in axs:
                self.set_format_to_ax(ax)
        else:
            for ax in axs.flat:
                self.set_format_to_ax(ax)

        return fig, axs


# class AxStyler:
#     """
#     Used to formt Axes
#     """
#     def __init__(self):
#         self.x_label = None
#         self.y_label = None
#         self.ax_title=None
#
#         self.grid = None


class GraphStyler:
    """
    Used, to format graphs.
    Call next_style, whenever you want to change current style.
    """

    def __init__(self, n_colors=4):
        self.cmap = plt.cm.viridis(np.linspace(0, 1, n_colors))
        self.marker_map = ["."]  # my_styles.MARKER_BERNHARD
        self.linestyle_map = [""]  # dict(linestyle='')

        self.marker_size = 4
        self.linewidth = 0
        self.marker_edge_width = 0
        self.current_color_index = 0
        self.current_marker_index = 0
        self.current_linestyle_index = 0

    @property
    def color(self) -> dict:
        assert 0 <= self.current_color_index < len(self.cmap), "current_color_index invalid!"
        return dict(color=self.cmap[self.current_color_index % len(self.cmap)])

    @property
    def marker(self) -> dict:
        assert 0 <= self.current_color_index < len(self.cmap), "current_marker_index invalid!"
        marker = self.marker_map[self.current_marker_index % len(self.marker_map)]
        return dict(marker=marker, markeredgewidth=self.marker_edge_width, markersize=self.marker_size)

    @property
    def linestyle(self) -> dict:
        assert 0 <= self.current_linestyle_index < len(self.linestyle_map), "current_linestyle_index invalid!"
        linestyle = self.linestyle_map[self.current_linestyle_index % len(self.linestyle_map)]
        return dict(linestyle=linestyle)

    @property
    def dict(self):
        result = self.color
        result.update(self.marker)
        result.update(self.linestyle)
        result.update(dict(linewidth=self.linewidth))
        return result

    def next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.cmap)

    def next_marker(self):
        self.current_marker_index = (self.current_marker_index + 1) % len(self.marker_map)

    def next_linestyle(self):
        self.current_linestyle_index = (self.current_linestyle_index + 1) % len(self.linestyle_map)

    def next_style(self):
        self.next_color()
        self.next_marker()
        self.next_linestyle()

    def reset(self):
        self.current_color_index = 0
        self.current_marker_index = 0
        self.current_linestyle_index = 0
        return self


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- graph styles ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_curve_style_bernhard_4(marker_size=5) -> GraphStyler:
    result = GraphStyler()
    result.linestyle_map = [""]
    result.marker_map = ["o", "s", "<", ">"]  # ["x", "+", "1"]
    result.marker_size = marker_size
    result.cmap = [[0, 0, 0],  # black
                   [1, 0, 0],  # red
                   [0, 0, 1],  # blue
                   [0, 1, 0]  # green
                   ]
    result.graph_styler = result
    return result


def get_power_law_fit_curve_style() -> GraphStyler:
    result = GraphStyler()
    result.linestyle_map = ["--"]
    result.marker_map = [""]  # ["x", "+", "1"]
    result.marker_size = 0
    result.cmap = [[1, 0.5, 0]]
    return result


# ---------------------------------------------------------------------------------
# -------------------------- plotter styls for AFM data ---------------------------
# ---------------------------------------------------------------------------------
def get_plotter_style_default(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = PlotterStyle(dpi=dpi, figure_size=figure_size)
    return result


def get_plotter_style_bernhard_4(dpi=300, figure_size=(5.6, 5.0), marker_size=5) -> PlotterStyle:
    """
    Max. 4 different graphs (black, red, blue, green)
    """
    result = PlotterStyle()
    result.set(dpi=dpi, figure_size=figure_size)
    graph_styler = GraphStyler()
    graph_styler.linestyle_map = [""]
    graph_styler.marker_map = ["o", "s", "<", ">"]  # ["x", "+", "1"]
    graph_styler.marker_size = marker_size
    graph_styler.cmap = [[0, 0, 0],  # black
                         [1, 0, 0],  # red
                         [0, 0, 1],  # blue
                         [0, 1, 0]  # green
                         ]
    result.graph_styler = graph_styler
    return result


def get_plotter_style_xy_data(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_default(dpi=dpi, figure_size=figure_size)
    result.graph_styler = get_curve_style_bernhard_4()
    return result


def get_plotter_style_rms(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_xy_data(dpi=dpi, figure_size=figure_size)
    result._x_label = "x"  # "[µm]"
    result._x_unit = "µm"
    result._y_label = "rms roughness"  # "[µm]"
    result._y_unit = "µm"

    return result


def get_plotter_style_sigma(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_xy_data(dpi=dpi, figure_size=figure_size)
    result._x_label = "[µm]"
    result._y_label = "standard deviation \u03C3 [µm]"
    return result
