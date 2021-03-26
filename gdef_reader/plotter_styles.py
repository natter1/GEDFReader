"""
@author: Nathanael Jöhrmann
"""
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


class PlotterStyle:
    def __init__(self, dpi: Optional[int] = None, figure_size: Optional[Tuple[float, float]] = None):
        self.dpi = dpi
        self.figure_size = figure_size

        self.graph_styler = None


class GraphStyler:
    """
    Used, to format graphs.
    Call next_style, whenever you want to change current style.
    """
    def __init__(self, n_colors=4):
        self.cmap = plt.cm.viridis(np.linspace(0, 1, n_colors))
        self.marker_map = ["."]  # my_styles.MARKER_BERNHARD
        self.linestyle_map = [""]  #dict(linestyle='')

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

# ---------------------------------------------------------------------------------
# ---------------------- some default styles for gdef_reader ----------------------
# ---------------------------------------------------------------------------------
def get_plotter_style_default() -> PlotterStyle:
    result = PlotterStyle(dpi=300, figure_size=(5.6, 5.0))
    return result


def get_plotter_style_bernhard_4(marker_size=5) -> PlotterStyle:
    """
    Max. 4 different graphs (black, red, blue, green)
    """
    result = PlotterStyle()
    graph_styler = GraphStyler()
    graph_styler.linestyle_map = [""]
    graph_styler.marker_map = ["o", "s", "<", ">"]  # ["x", "+", "1"]
    graph_styler.marker_size = marker_size
    graph_styler.cmap = [[0, 0, 0], # black
                         [1, 0, 0],  # red
                         [0, 0, 1],  # blue
                         [0, 1, 0]  # green
                        ]
    result.graph_styler = graph_styler
    return result


def get_power_law_fit_curve_style() -> GraphStyler:
    result = GraphStyler()
    result.linestyle_map = ["--"]
    result.marker_map = [""]  # ["x", "+", "1"]
    result.marker_size = 0
    result.cmap = [[1, 0.5, 0]]
    return result