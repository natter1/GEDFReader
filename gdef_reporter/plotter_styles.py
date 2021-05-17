"""
@author: Nathanael Jöhrmann
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# [0, 1, 1]  # define color by float rgb value
from matplotlib.colors import to_rgba

CMAP_BERNHARD = ["black", "red", "blue", "limegreen", "aqua", "magenta", "orange", "sienna", "gold", "purple"]
MARKER_BERHNHARD = ["o", "s", "<", ">", "D", "P", "X", "*"]  # , "x", "+", "1"]


class PlotterStyle:
    dpi: int
    figure_size: tuple[float, float]
    fig_title: Optional[str]
    ax_title: Optional[str]
    grid: Optional[bool]

    def __init__(self, dpi: Optional[int] = None, figure_size: Optional[Tuple[float, float]] = None):
        self.dpi = dpi
        self.figure_size = figure_size
        self.fig_title = None

        self._x_label = None
        self._y_label = None
        self._x_unit = None
        self._y_unit = None
        self.ax_title = None
        self.grid = None

        # self.ax_styler = AxStyler()
        self.graph_styler = None
        self.hist_styler = None

    @property
    def x_label(self) -> Optional[str]:
        if self._x_label is None and self._x_unit is None:
            return None
        result = ""
        if self._x_label:
            result += self._x_label
        if self._x_unit is not None:
            result += f" [{self._x_unit}]"
        return result

    @property
    def y_label(self) -> Optional[str]:
        if self._y_label is None and self._y_unit is None:
            return None
        result = ""
        if self._y_label:
            result += self._y_label
        if self._y_unit:
            result += f" [{self._y_unit}]"
        return result

    @property
    def x_unit(self):
        return self._x_unit

    @property
    def y_unit(self):
        return self._y_unit

    def set(self, dpi=None, figure_size=None, x_label=None, y_label=None,
            x_unit=None, y_unit=None,
            fig_title=None, ax_title=None,
            grid=None) -> None:
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

        if fig_title is not None:
            self.fig_title = fig_title
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
        fig, axs = plt.subplots(nrows, ncols, figsize=self.figure_size, dpi=self.dpi, tight_layout=True)
        if self.fig_title:
            fig.suptitle(self.fig_title)
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


class BaseStyler(ABC):
    """
    Abstract base class for GraphStyler, HistStyler ... Provides basic color handling.
    """
    def __init__(self, n_colors):
        self.cmap = plt.cm.viridis(np.linspace(0, 1, n_colors))
        self.current_color_index = 0

    def _color(self) -> dict:
        assert 0 <= self.current_color_index < len(self.cmap), "current_color_index invalid!"
        return self.cmap[self.current_color_index % len(self.cmap)]

    @abstractmethod
    def dict(self):
        pass

    @abstractmethod
    def next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.cmap)

    @abstractmethod
    def next_style(self):
        self.next_color()

    @abstractmethod
    def reset(self):
        self.current_color_index = 0


class GraphStyler(BaseStyler):
    """
    Used, to format graphs.
    Call next_style, whenever you want to change current style.
    """
    def __init__(self, n_colors=4):
        super().__init__(n_colors)  # cmap and color_index
        self.marker_map = ["."]  # my_styles.MARKER_BERNHARD
        self.linestyle_map = [""]  # dict(linestyle='')

        self.marker_size = 4
        self.linewidth = 0
        self.marker_edge_width = 0

        self.current_marker_index = 0
        self.current_linestyle_index = 0

    @property
    def color(self) -> dict:
        return dict(color=self._color())

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
        super().next_color()

    def next_marker(self):
        self.current_marker_index = (self.current_marker_index + 1) % len(self.marker_map)

    def next_linestyle(self):
        self.current_linestyle_index = (self.current_linestyle_index + 1) % len(self.linestyle_map)

    def next_style(self):
        super().next_color()
        self.next_marker()
        self.next_linestyle()

    def reset(self):
        super().reset()
        self.current_marker_index = 0
        self.current_linestyle_index = 0
        return self


class HistStyler(BaseStyler):
    """
    Used, to format histogram bars.
    Call next_style, whenever you want to change current style.
    """
    def __init__(self, n_colors=4):
        super().__init__(n_colors)
        self.histtype = "bar"
        self.fill = True
        self.density = True  # switch between counts and density for y-axis
        self.nbins = 50

        self.fc_alpha = 0.3  # alpha for fill color
        self.lw = 0.5  # line-width of bar edges
        self.rwidth = 1.0  # width of bars (0.0 ... 1.0)

    @property
    def color(self) -> dict:
        return dict(edgecolor=self._color(), fc=to_rgba(self._color(), alpha=self.fc_alpha))

    @property
    def histogram(self) -> dict:
        return dict(histtype=self.histtype, fill=self.fill, density=self.density, bins=self.nbins)

    @property
    def bar(self) -> dict:
        return dict(lw=self.lw, rwidth=self.rwidth)

    @property
    def dict(self):
        result = self.color
        result.update(self.histogram)
        result.update(self.bar)
        return result

    def next_color(self):
        super().next_color()

    def next_style(self):
        super().next_color()

    def reset(self):
        super().reset()

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- graph styles ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_graph_style_bernhard(marker_size=5) -> GraphStyler:
    result = GraphStyler()
    result.linestyle_map = [""]
    result.marker_map = MARKER_BERHNHARD
    result.marker_size = marker_size
    result.cmap = CMAP_BERNHARD
    return result


def get_power_law_fit_curve_style() -> GraphStyler:
    result = GraphStyler()
    result.linestyle_map = ["--"]
    result.marker_map = [""]  # ["x", "+", "1"]
    result.marker_size = 0
    result.cmap = [[1, 0.5, 0]]
    return result


# ---------------------------------------------------------------------------------
# ----------------------------- histogram styles for ------------------------------
# ---------------------------------------------------------------------------------
def get_histogram_style_bernhard(nbins=50) -> HistStyler:
    result = HistStyler()
    result.cmap = CMAP_BERNHARD
    result.nbins=nbins
    return result


# ---------------------------------------------------------------------------------
# ------------------------- plotter styles for AFM data ---------------------------
# ---------------------------------------------------------------------------------
def get_plotter_style_default(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = PlotterStyle(dpi=dpi, figure_size=figure_size)
    return result


def get_plotter_style_bernhard(dpi=300, figure_size=(5.6, 5.0), marker_size=5, nbins=50) -> PlotterStyle:
    """
    "black", "red", "blue", "limegreen", "aqua", "magenta", "orange", "sienna", "gold", "purple"
    """
    result = PlotterStyle()
    result.set(dpi=dpi, figure_size=figure_size)
    result.graph_styler = get_graph_style_bernhard(marker_size=marker_size)
    result.hist_styler = get_histogram_style_bernhard(nbins=nbins)
    return result


def get_plotter_style_histogram(dpi=300, figure_size=(5.6, 5.0), nbins=50) -> PlotterStyle:
    result = get_plotter_style_bernhard(dpi=dpi, figure_size=figure_size)
    result.set(x_label="z", y_label="Normalized counts")
    result.hist_styler.nbins = nbins
    return result


def get_plotter_style_xy_data(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_bernhard(dpi=dpi, figure_size=figure_size)
    return result


def get_plotter_style_rms(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_xy_data(dpi=dpi, figure_size=figure_size)
    result.set(x_label="x", x_unit="µm", y_label="rms roughness", y_unit="nm")
    return result


def get_plotter_style_sigma(dpi=300, figure_size=(5.6, 5.0)) -> PlotterStyle:
    result = get_plotter_style_xy_data(dpi=dpi, figure_size=figure_size)
    result.set(x_label="x", x_unit="µm", y_label="standard deviation \u03C3", y_unit="µm")
    return result
